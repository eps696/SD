# from https://github.com/AmericanPresidentJimmyCarter/stable-diffusion/blob/main/src/stable_inference/sampling.py

import os, sys
import re
from typing import Iterable
from functools import partial
import urllib.request

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'xtra'))
import k_diffusion as K

from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.modules.embedding_manager import EmbeddingManager

from transformers import logging
logging.set_verbosity_error() # stop transformers warning

DYNAMIC_THRESHOLDING_MIMIC_SCALE = 7.5
DYNAMIC_THRESHOLDING_PERCENTILE = 0.9995

TAGS_RE = re.compile('<.*?>')
sd_concepts_url_fn = lambda concept: f'https://huggingface.co/sd-concepts-library/{concept}/resolve/main/'
UNLIKELY_TOKENS = ['?', '°', '±', '?', '?', '?', 'µ', '·', '?', '?', '?', '»', '?', '?', '?',]

def prompt_injects(prompt, embedding_dir, model_dir='models', use_half=True):
    ''' Inject custom concept into a prompt '''
    def _next_token_for_concept():
        for token in UNLIKELY_TOKENS:
            yield token
        yield None

    def _get_clip_token_for_string(tokenizer, string):
        batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, \
                                   padding='max_length', return_tensors='pt')
        tokens = batch_encoding['input_ids']
        if torch.count_nonzero(tokens - 49407) == 2:
            return tokens[0, 1]
        return None

    def _get_placeholder_loop(embedder):
        new_placeholder  = None
        token_seq = _next_token_for_concept()
        while True:
            new_placeholder = next(token_seq)
            if new_placeholder is None:
                raise ValueError('Ran out of tokens due to too many ' + f'concepts (max: {len(UNLIKELY_TOKENS)})')
            token = _get_clip_token_for_string(embedder.tokenizer, new_placeholder)
            if token is not None:
                return new_placeholder, token

    if isinstance(prompt, list): prompt = ' | '.join(prompt) # injection works with single prompts
    prompt_injected = prompt
    embedding_manager = None

    if '<' in prompt and '>' in prompt:
        embedding_paths = []
        for tag in re.findall(TAGS_RE, prompt):
            concept = tag[1:-1]
            concept_file_path = os.path.join(embedding_dir, f'{concept}.pt')

            if not os.path.isfile(concept_file_path): # no local pt, trying bin from hf
                print('not found', concept_file_path)
                tag_actual = None
                token_name_path = os.path.join(embedding_dir, f'{concept}/token_identifier.txt')
                concept_file_path = os.path.join(embedding_dir, f'{concept}/learned_embeds.bin')
                if not os.path.isfile(token_name_path): # not found, downloading
                    try:
                        os.makedirs(os.path.join(embedding_dir, concept), exist_ok=True)
                        urllib.request.urlretrieve(sd_concepts_url_fn(concept) + 'token_identifier.txt', token_name_path)
                        urllib.request.urlretrieve(sd_concepts_url_fn(concept) + 'learned_embeds.bin', concept_file_path)
                    except:
                        print(' Inversion embeddings not found anywhere! Ignoring..')
                if not os.path.isfile(token_name_path): # not found on the web
                    return prompt_injected, None

                with open(token_name_path, 'r') as token_name_file:
                    tag_actual = token_name_file.read()
                prompt_injected = prompt_injected.replace(tag, tag_actual)

            embedding_paths.append(concept_file_path)

        # Merge the embeddings.
        embedder = FrozenCLIPEmbedder(model_dir=model_dir).cuda()
        EmbeddingManagerCls = partial(EmbeddingManager, embedder, ["*"])

        string_to_token_dict = {}
        string_to_param_dict = torch.nn.ParameterDict()
        placeholder_to_src = {}

        for manager_ckpt in embedding_paths:
            manager = EmbeddingManagerCls()
            manager.load(manager_ckpt)
            if use_half: manager = manager.half()

            for placeholder_string in manager.string_to_token_dict:
                if not placeholder_string in string_to_token_dict:
                    string_to_token_dict[placeholder_string] = manager.string_to_token_dict[placeholder_string]
                    string_to_param_dict[placeholder_string] = manager.string_to_param_dict[placeholder_string]
                    placeholder_to_src[placeholder_string] = manager_ckpt
                else:
                    new_placeholder, new_token = _get_placeholder_loop(embedder)
                    string_to_token_dict[new_placeholder] = new_token
                    string_to_param_dict[new_placeholder] = manager.string_to_param_dict[placeholder_string]
                    placeholder_to_src[new_placeholder] = manager_ckpt

        merged_manager = EmbeddingManagerCls()
        if use_half: manager = manager.half()
        merged_manager.string_to_param_dict = string_to_param_dict
        merged_manager.string_to_token_dict = string_to_token_dict
        embedding_manager = merged_manager

    return prompt_injected, embedding_manager


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class KCFGDenoiser(torch.nn.Module):
    ''' k-diffusion sampling with multi-conditionable denoising '''
    def __init__(self, model, scale_factor: float):
        super().__init__()
        self.inner_model = model
        self.scale_factor = scale_factor

    def forward(self, x, sigma, uncond, cond, cond_scale, cond_counts=None, cond_weights=None, dynamic_threshold=True, use_half=False, mask=None, x_frozen=None):
        ''' Magicool k-sampler prompt positive/negative weighting from birch-san.
        https://github.com/Birch-san/stable-diffusion/blob/birch-mps-waifu/scripts/txt2img_fork.py
        '''
        device = x.device
        uncond_count = -1
        if not isinstance(uncond, dict):
            uncond_count = uncond.size(dim=0)
        else:
            uncond_count = uncond['c_crossattn'][0].size(dim=0)
        if not isinstance(cond, dict):
            cond_count = cond.size(dim=0)
        else:
            cond_count = cond['c_crossattn'][0].size(dim=0)

        cond_counts_tensor = torch.tensor(cond_counts, device=device)
        if isinstance(cond, dict):
            assert isinstance(uncond, dict)
            cond_in = dict()
            for k in cond:
                if isinstance(cond[k], list) and k != 'c_concat':
                    cond_in[k] = [torch.cat([uncond[k][i], cond[k][i]]).to(device) for i in range(len(cond[k]))]
                elif isinstance(cond[k], list) and k == 'c_concat':
                    # TODO This might be wrong if cond and uncond c_concat tensors are different, but with the RML inpainting model they are not.
                    # first spread refers to when empty c_concat are used with hybrid conditioning, while the second spread is for properly generated image conditions
                    spread = 1 + cond_count
                    if cond[k][0].size()[0] > 1:
                        spread = 3 + (cond_count - 8) // 4
                    cond_in[k] = [torch.tile(cond[k][i], (spread, 1, 1, 1)) for i in range(len(cond[k]))]
                else:
                    cond_in[k] = torch.cat([uncond[k], cond[k]]).to(device)
        else:
            cond_in = torch.cat([uncond, cond]).to(device)

        del uncond, cond
        if use_half and (x.dtype == torch.float32 or x.dtype == torch.float64):
            x = x.half()
        x_in = cat_self_with_repeat_interleaved(t=x, factors_tensor=cond_counts_tensor, factors=cond_counts, output_size=cond_count)
        del x
        sigma_in = cat_self_with_repeat_interleaved(t=sigma, factors_tensor=cond_counts_tensor, factors=cond_counts, output_size=cond_count)
        del sigma
        uncond_out, conds_out = self.inner_model(x_in, sigma_in, cond=cond_in).split([uncond_count, cond_count])
        del x_in, sigma_in, cond_in
        unconds = repeat_interleave_along_dim_0(t=uncond_out, factors_tensor=cond_counts_tensor, factors=cond_counts, output_size=cond_count)
        del cond_counts_tensor
        # transform
        #   tensor([0.5, 0.1])
        # into:
        #   tensor([[[[0.5000]]],
        #           [[[0.1000]]]])
        weight_tensor = torch.tensor(list(cond_weights), device=device, dtype=uncond_out.dtype) * cond_scale
        weight_tensor = weight_tensor.reshape(len(list(cond_weights)), 1, 1, 1)
        deltas: torch.Tensor = (conds_out - unconds)

        if dynamic_threshold:
            deltas_target = deltas * weight_tensor
            weight_tensor_mimic = torch.tensor(list(map(lambda weight: weight / max(cond_weights), cond_weights)), device=device, dtype=uncond_out.dtype) \
                                               * DYNAMIC_THRESHOLDING_MIMIC_SCALE
            weight_tensor_mimic = weight_tensor_mimic.reshape(len(list(cond_weights)), 1, 1, 1)
            deltas_mimic = deltas * weight_tensor_mimic
            deltas = dynamic_thresholding_of_image_latent(deltas_target, deltas_mimic, self.scale_factor)
        else:
            deltas = deltas * weight_tensor
        del conds_out, unconds, weight_tensor
        cond = sum_along_slices_of_dim_0(deltas, counts=cond_counts)
        del deltas
        denoised = uncond_out + cond # latent_samples
        if mask is not None:
            assert x_frozen is not None
            img_orig = x_frozen
            mask_inv = 1. - mask
            denoised = (img_orig * mask) + (denoised * mask_inv)

        return denoised

def cat_self_with_repeat_interleaved(t: torch.Tensor, factors: Iterable[int], factors_tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    """ Fast-paths for a pattern which in its worst-case looks like:
    torch.cat((t, t.repeat_interleave(factors, dim=0)))
    t = torch.tensor([[0,1],[2,3]]); factors=(2,3)
    = tensor([[0,1],[2,3],[0,1],[0,1],[2,3],[2,3],[2,3]])
    Fast-path:
      `len(factors) == 1`
      it's just a normal repeat
    t = torch.tensor([[0,1]]); factors=(2)
    = tensor([[0,1],[0,1],[0,1]])
    t = torch.tensor([[0,1],[2,3]]); factors=(2)
    = tensor([[0,1],[2,3],[0,1],[2,3],[0,1],[2,3]])
    """
    if len(factors) == 1:
        return repeat_along_dim_0(t, factors[0]+1)
    return torch.cat((t, repeat_interleave_along_dim_0(t=t, factors_tensor=factors_tensor, factors=factors, output_size=output_size))).to(t.device)

def repeat_along_dim_0(t: torch.Tensor, factor: int) -> torch.Tensor:
    """ Repeats a tensor's contents along its 0th dim `factor` times.
    repeat_along_dim_0(torch.tensor([[0,1]]), 2)
    = tensor([[0,1],[0,1]])
    # shape changes from (1,2) to (2,2)
    repeat_along_dim_0(torch.tensor([[0,1],[2,3]]), 2)
    = tensor([[0,1],[2,3],[0,1],[2,3]])
    # shape changes from (2,2) to (4,2)
    """
    assert factor >= 1
    if factor == 1:
        return t
    if t.size(dim=0) == 1:
        # prefer expand() whenever we can, since doesn't copy
        return t.expand(factor * t.size(dim=0), *(-1,)*(t.ndim-1))
    return t.repeat((factor, *(1,)*(t.ndim-1)))

def repeat_interleave_along_dim_0(t: torch.Tensor, factors: Iterable[int], factors_tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    """ repeat_interleave()s a tensor's contents along its 0th dim.
    factors = (2,3)
    factors_tensor = torch.tensor(factors)
    output_size = factors_tensor.sum().item() # 5
    t = torch.tensor([[0,1],[2,3]])
    repeat_interleave_along_dim_0(t=t, factors=factors, factors_tensor=factors_tensor, output_size=output_size)
    = tensor([[0,1],[0,1],[2,3],[2,3],[2,3]])
    """
    factors_len = len(factors)
    assert factors_len >= 1
    if len(factors) == 1:
        # prefer repeat() whenever we can, because MPS doesn't support repeat_interleave()
        return repeat_along_dim_0(t, factors[0])
    if t.device.type != 'mps':
        return t.repeat_interleave(factors_tensor, dim=0, output_size=output_size)
    return torch.cat([repeat_along_dim_0(split, factor)
        for split, factor in zip(t.split(1, dim=0), factors)]).to(t.device)

def sum_along_slices_of_dim_0(t: torch.Tensor, counts: Iterable[int]) -> torch.Tensor:
    """ Implements fast-path for a pattern which in the worst-case looks like this:
    t = torch.tensor([[1],[2],[3]]); counts = (2,1)
    torch.cat([torch.sum(split, dim=0, keepdim=True) for split in t.split(counts)])
    = tensor([[3],[3]])
    Fast-path:
      `len(counts) == 1`
      it's just a normal sum(t, dim=0, keepdim=True)
    t = torch.tensor([[1],[2]]); counts=(2)
    t.sum(dim=0, keepdim=True)
    = tensor([[3]])
    """
    if len(counts) == 1:
        if t.size(dim=0) == 1:
            return t
        return t.sum(dim=0, keepdim=True)
    splits: List[torch.Tensor] = t.split(counts)
    device = t.device
    del t
    sums: List[torch.Tensor] = [torch.sum(split, dim=0, keepdim=True) for split in splits]
    del splits
    return torch.cat(sums).to(device)

def dynamic_thresholding_of_image_latent(
    latent_images: torch.Tensor,
    latent_images_target_scaling: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    ''' One of birch-san's methods for dynamic thresholding '''
    dt_unscaled: torch.Tensor = latent_images_target_scaling / scale_factor
    dt_flattened: torch.Tensor = dt_unscaled.flatten(2)
    dt_means: torch.Tensor = dt_flattened.mean(dim=2).unsqueeze(2)
    dt_recentered: torch.Tensor = dt_flattened - dt_means
    dt_q = torch.quantile(dt_recentered.abs(), DYNAMIC_THRESHOLDING_PERCENTILE, dim=2)

    ut_unscaled:  torch.Tensor = latent_images / scale_factor
    ut_flattened:  torch.Tensor = ut_unscaled.flatten(2)
    ut_means:  torch.Tensor = ut_flattened.mean(dim=2).unsqueeze(2)
    ut_centered:  torch.Tensor = ut_flattened - ut_means

    ut_abs = ut_centered.abs()
    ut_q = torch.quantile(ut_abs, DYNAMIC_THRESHOLDING_PERCENTILE, dim=2)
    ut_q = torch.maximum(ut_q, dt_q)
    q_ratio = ut_q / dt_q
    q_ratio = q_ratio.unsqueeze(2).expand(*ut_centered.shape)
    t_scaled = ut_centered / q_ratio

    uncentered: torch.Tensor = t_scaled + ut_means
    unflattened: torch.Tensor = uncentered.unflatten(2, latent_images_target_scaling.shape[2:])
    scaled = unflattened * scale_factor

    return scaled


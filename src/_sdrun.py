import os, sys
import time
import argparse

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

import logging
logging.basicConfig(level=logging.ERROR) # stop xformers warnings
import transformers
transformers.logging.set_verbosity_error() # stop transformers warning

sys.path.append(os.path.join(os.path.dirname(__file__), 'xtra'))
import k_diffusion as K
from ldm.models.diffusion.ddim import DDIMSampler
from sampling import KCFGDenoiser, prompt_injects
from utils import load_model, load_img, save_img, makemask, calc_size, txt_clean, read_txt, read_multitext, precision_cast, prep_midas, log_tokens, img_list, basename, progbar
# try: # progress bar for notebooks 
    # get_ipython().__class__.__name__
    # from util.progress_bar import ProgressIPy as progbar
# except: # normal console
    # from util.progress_bar import ProgressBar as progbar

samplers = ['ddim', 'klms', 'euler', 'euler_a', 'dpm_ada', 'dpm_fast', 'dpm2_a'] # 'heun', 'dpmpp_2s_a', 'dpm2', 'dpmpp_2m'
models = { 
    '15'  : ['src/yaml/v1-inference.yaml',   'models/sd-v15-512-fp16.ckpt',         512],
    '15i' : ['src/yaml/v1-inpainting.yaml',  'models/sd-v15-512-inpaint-fp16.ckpt', 512],
    'v2i' : ['src/yaml/v2-inpainting.yaml',  'models/sd-v2-512-inpaint-fp16.ckpt',  512],
    'v2d' : ['src/yaml/v2-midas.yaml',       'models/sd-v2-512-depth-fp16.ckpt',    512],
    'v21' : ['src/yaml/v2-inference.yaml',   'models/sd-v21-512-fp16.ckpt',         512],
    'v21v': ['src/yaml/v2-inference-v.yaml', 'models/sd-v21v-768-fp16.ckpt',        768],
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--in_txt',   default=None, help='Text string or file to process')
    parser.add_argument('-pre', '--prefix', default=None, help='Prefix for input text')
    parser.add_argument('-post', '--postfix', default=None, help='Postfix for input text')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images (supersedes width and height)')
    parser.add_argument('-M', '--mask',     default=None, help='Path to input mask for inpainting mode (supersedes width and height)')
    parser.add_argument('-inv', '--invert_mask', action='store_true')
    parser.add_argument('-em', '--embeds',  default='_in/embed', help='Path to directories with embeddings')
    parser.add_argument('-o', '--out_dir',  default="_out", help="Output directory for generated images")
    parser.add_argument('-md', '--maindir', default='./', help='Main SD directory')
    parser.add_argument('-sz', '--size',    default=None, help="image width, multiple of 32")
    parser.add_argument('-m',  '--model',   default='15', choices=models.keys(), help="model version [15,15i,v2i,v2d,v21,v21v]")
    parser.add_argument('-sm', '--sampler', default='euler', choices=samplers)
    parser.add_argument('-nz', '--noiser',  default='default', choices=['default', 'karras', 'exponential', 'vp'])
    parser.add_argument(       '--vae',     default='ema', help='orig, ema, mse')
    parser.add_argument('-C','--cfg_scale', default=7.5, type=float, help="prompt configuration scale")
    parser.add_argument('-f', '--strength', default=0.75, type=float, help="strength for noising/unnoising. 0 = preserve img, 1 = replace it completely")
    parser.add_argument('-s', '--steps',    default=50, type=int, help="number of diffusion steps")
    parser.add_argument('--precision',      default='autocast')
    parser.add_argument('-S','--seed',      type=int, help="image seed")
    parser.add_argument('-v', '--verbose',  action='store_true')
    return parser.parse_args()

SIGMA_MIN = 0.0292
SIGMA_MAX = 14.6146
device = torch.device('cuda')

def sd_setup(a):
    model = load_model(*[os.path.join(a.maindir, d) for d in models[a.model][:2]], a.maindir)
    if model.parameterization == "v":
        try:
            import xformers
        except:
            print(" V-models require xformers! install xformers or use another model"); exit()
        model_wrap = K.external.CompVisVDenoiser(model)
    else:
        model_wrap = K.external.CompVisDenoiser(model)
    model_cfg = KCFGDenoiser(model_wrap, scale_factor=model.scale_factor)

    a.hybrid_cond = model.uses_rml_inpainting or hasattr(model, 'depth_model') # runwayml inpaint or depth
    if a.hybrid_cond is True: a.sampler = 'ddim'
    sampler = DDIMSampler(model, device=device)
    sampler.make_schedule(ddim_num_steps=a.steps, ddim_eta=0., ddim_discretize='uniform', verbose=False)

    if a.sampler == 'klms': # fast, consistent for interpolation
        sampling_fn = K.sampling.sample_lms
    elif a.sampler == 'euler': # fast
        sampling_fn = K.sampling.sample_euler
    elif a.sampler == 'euler_a': # fast
        sampling_fn = K.sampling.sample_euler_ancestral
    elif a.sampler == 'dpm_ada': # fast
        sampling_fn = lambda *args, **kwargs: K.sampling.sample_dpm_adaptive(args[0], args[1], SIGMA_MIN, SIGMA_MAX, **kwargs)
    elif a.sampler == 'dpm_fast': 
        sampling_fn = lambda *args, **kwargs: K.sampling.sample_dpm_fast(args[0], args[1], SIGMA_MIN, SIGMA_MAX, a.steps, **kwargs)
    elif a.sampler == 'dpm2_a': # slow, a la euler_a
        sampling_fn = K.sampling.sample_dpm_2_ancestral
    # elif a.sampler == 'heun': # slow, a la euler
        # sampling_fn = K.sampling.sample_heun
    # elif a.sampler == 'dpm2': # slow, like heun
        # sampling_fn = K.sampling.sample_dpm_2
    # elif a.sampler == 'dpmpp_2m': # fast, like klms
        # sampling_fn = K.sampling.sample_dpmpp_2m
    # elif a.sampler == 'dpmpp_2s_a': # slow, a la dpm2_a
        # sampling_fn = K.sampling.sample_dpmpp_2s_ancestral

    sigmas = model_wrap.get_sigmas(a.steps) # num = a.steps + 1
    if 'karras' in a.noiser:
        sigmas = K.sampling.get_sigmas_karras(a.steps, SIGMA_MIN, SIGMA_MAX, rho=7., device=device)
    elif 'expo' in a.noiser:
        sigmas = K.sampling.get_sigmas_exponential(a.steps, SIGMA_MIN, SIGMA_MAX, device=device)
    elif 'vp' in a.noiser:
        sigmas = K.sampling.get_sigmas_vp(a.steps, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device=device)

    # only for images
    t_enc = int(a.strength * a.steps)
    sigma_lat = sigmas[len(sigmas) - t_enc - 1]
    if hasattr(a, 'in_img') and a.in_img is not None:
        sigmas = sigmas[len(sigmas) - t_enc - 1 :]

    uc = model.get_learned_conditioning([""])
    if a.seed is None: a.seed = int((time.time()%1)*69696)

    if a.model.startswith('1') and a.vae != 'orig':
        vae = 'vae-ft-mse-840000.ckpt' if a.vae=='mse' else 'vae-ft-ema-560000.ckpt'
        vae_ckpt = torch.load(os.path.join(a.maindir, 'models', vae), map_location="cpu")
        vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}
        model.first_stage_model.load_state_dict(vae_dict, strict=False)

    def img_lat(image):
        if a.precision not in ['full', 'float32']: image = image.half()
        return model.get_first_stage_encoding(model.encode_first_stage(image)) # move to latent space
    def lat_z(lat): # for k-samplers
        return lat + torch.randn_like(lat) * sigma_lat
    def lat_z_enc(lat): # for ddim sampler
        return sampler.stochastic_encode(lat, torch.tensor([t_enc]).to(device))
    def img_z(image):
        return lat_z_enc(img_lat(image)) if a.sampler=='ddim' else lat_z(img_lat(image))
    def rnd_z(H, W):
        return torch.randn([1, model.channels, H, W], device=device) * sigmas[0] # [1,4,64,64] noise
    def txt_c(txt):
        args = [txt] if a.embeds is None else prompt_injects(txt, a.embeds)
        return model.get_learned_conditioning(*args)

    @precision_cast
    def generate(z_, c_, uc=uc, cw=None, img=None, mask=None, thresh=True):
        with model.ema_scope():
            if a.sampler == 'ddim': # ddim decode = required for hybrid_cat [depth/inpaint]
                samples = sampler.decode(z_, c_, t_enc, unconditional_guidance_scale=a.cfg_scale, unconditional_conditioning=uc) # [1,4,64,64]
            else:
                extra_args = {'cond': c_, 'uncond': uc, 'cond_scale': a.cfg_scale, 'dynamic_threshold': thresh} # thresholding fails if c=uc!
                if mask is not None and img is not None: 
                    extra_args = {**extra_args, 'x_frozen': img, 'mask': mask}
                c_count = c_.shape[0]
                if cw is None: cw = [1.] * c_count
                extra_args['cond_weights'] = [c / sum(cw) for c in cw]
                extra_args['cond_counts'] = [c_count,]
                samples = sampling_fn(model_cfg, z_, sigmas, extra_args=extra_args, disable=False) # [1,4,64,64]
            return model.decode_first_stage(samples)[0] # [3,h,w]

    return [a, model, uc, img_lat, lat_z, lat_z_enc, img_z, rnd_z, txt_c, generate]

def main():
    # main setup
    a = get_args()
    [a, model, uc, img_lat, lat_z, lat_z_enc, img_z, rnd_z, txt_c, generate] = sd_setup(a)
    seed_everything(a.seed)

    posttxt = basename(a.in_txt) if a.in_txt is not None and os.path.exists(a.in_txt) else ''
    postimg = basename(a.in_img) if a.in_img is not None and os.path.exists(a.in_img) else ''
    if len(posttxt) > 0 or len(postimg) > 0:
        a.out_dir = os.path.join(a.out_dir, posttxt + '-' + postimg)
        a.out_dir += '-' + a.model
    os.makedirs(a.out_dir, exist_ok=True)

    size = None if a.size is None else calc_size(a.size, a.model, a.verbose) 
    if model.uses_rml_inpainting and a.mask is None:
        print('!! inpainting models need mask !!'); exit()
    if a.verbose: print('..', basename(models[a.model][1]), '..', a.sampler, '..', a.strength)

    prompts, cws = read_multitext(a.in_txt, a.prefix, a.postfix, flat=a.hybrid_cond)
    count = len(prompts)

    if a.in_img is not None and os.path.exists(a.in_img):
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        count = max(count, len(img_paths))

    if a.mask is not None: 
        masks = img_list(a.mask) if os.path.isdir(a.mask) else read_txt(a.mask)

    pbar = progbar(count)
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        # if a.verbose: log_tokens(' '.join(prompt).strip(), model)
        c_ = txt_c(prompt)

        # img2img
        if a.in_img is not None and os.path.exists(a.in_img):
            img_path = img_paths[i % len(img_paths)]
            file_out = os.path.basename(img_path)
            init_image, (W,H) = load_img(img_path, size)
            lat_ = img_lat(init_image)
            mask = None if a.mask is None else makemask(img_path, masks[i % len(masks)], a.invert_mask, resize = not a.hybrid_cond)

            if a.hybrid_cond: # depth/inpaint models, ddim sampler

                # img2img with depth model
                if hasattr(model, 'depth_model'): 
                    with torch.no_grad(), torch.autocast("cuda"):
                        dd = model.depth_model(prep_midas(init_image)) # [1,1,384,384]
                    dd = torch.nn.functional.interpolate(dd, size=[H//8, W//8], mode="bicubic", align_corners=False)
                    depth_min, depth_max = torch.amin(dd, dim=[1, 2, 3], keepdim=True), torch.amax(dd, dim=[1, 2, 3], keepdim=True)
                    dd = 2. * (dd - depth_min) / (depth_max - depth_min) - 1.
                    hybrid_cat = torch.cat([dd]) # [1,1,64,64]

                # inpainting with runwayml model
                elif model.uses_rml_inpainting: 
                    masked = img_lat(init_image * mask) # [1,4,64,64]
                    mask = 1. - F.interpolate(mask, size=[H//8, W//8]) # [1,1,64,64]
                    hybrid_cat = torch.cat([mask, masked], dim=1)

                c_cat  = {"c_concat": [hybrid_cat], "c_crossattn": [c_]}
                uc_cat = {"c_concat": [hybrid_cat], "c_crossattn": [uc]}
                z_ = lat_z_enc(lat_)
                image = generate(z_, c_cat, uc_cat)

            else: # normal models, k-sampler
                z_ = lat_z(lat_)
                image = generate(z_, c_, cw=cws[i%len(cws)], img=lat_, mask=mask)

        # txt2img, k-sampler
        else:
            assert a.sampler != 'ddim', " Wrong sampler! Use k-samplers for text-only generation"
            file_out = '%s-%s-%s-%d.jpg' % (txt_clean(prompt)[:128], a.model, a.sampler, a.seed)
            W, H = [models[a.model][2]]*2 if size is None else size
            z_ = rnd_z(H//8, W//8) # [1,4,64,64] noise
            image = generate(z_, c_, cw=cws[i%len(cws)])
            
        save_img(image, 0, a.out_dir, filepath=file_out)
        pbar.upd()


if __name__ == '__main__':
    main()

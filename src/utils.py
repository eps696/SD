import os, sys, io
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
from typing import Callable
from PIL import Image, ImageOps
import cv2
import skimage

import torch

from ldm.util import instantiate_from_config
from ldm.modules.midas.api import load_midas_transform
from txt2mask import Txt2Mask

device = torch.device('cuda')

def gpu_ram():
    return torch.cuda.get_device_properties(0).total_memory // (1024*1024*1023)

def calc_size(size, model, verbose=True):
    if size.lower() == 'max': # if no xformers
        gpuram = gpu_ram()
        if model[-1]=='v':
            size = '768' if gpuram < 12 else '928-768'
        else:
            size = '512-448' if gpuram < 8 else '704-512' if gpuram < 10 else '896-512' if gpuram < 12 else '1024-576'
        if verbose: print('GPU RAM', gpuram, 'resolution', size)
    size = [int(s) for s in size.split('-')]
    if len(size)==1: size = size * 2
    w, h = map(lambda x: x - x % 8, size)  # resize to integer multiple of 8
    return w, h

def lerp(v0, v1, x):
    return (1.-x) * v0 + x * v1

def lerps(v0, v1, x): # for lists
    assert len(v0) == len(v1)
    return [(1.-x) * x0 + x * x1 for x0,x1 in zip(v0,v1)]

# https://github.com/itjustis/sdthings/blob/main/scripts/alivify.py - from @xsteenbrugge?
def slerp(v0, v1, x, DOT_THRESHOLD=0.9995):
    dot = torch.sum(v0 * v1 / (torch.norm(v0) * torch.norm(v1)))
    if torch.abs(dot) > DOT_THRESHOLD:
        v2 = lerp(v0, v1, x)
    else:
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * x
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2

def load_model(cfg_path, model_path, maindir, embedding=None, use_half=True):
    cfg = OmegaConf.load(cfg_path)
    cfg['model']['params']['cond_stage_config']['params'] = {'model_dir': os.path.join(maindir, 'models')} # to FrozenCLIPEmbedder
    with open(model_path,'rb') as f:
        model_bytes = f.read()
    pl_sd = torch.load(io.BytesIO(model_bytes), map_location='cpu')
    del model_bytes
    model = instantiate_from_config(cfg.model)
    m, u  = model.load_state_dict(pl_sd['state_dict'], strict=False)
    if use_half is True:
        model.to(torch.float16)
    model.to(device)
    model.cond_stage_model.device = device
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            m._orig_padding_mode = m.padding_mode
    if embedding is not None:
        model.embedding_manager.load(embedding, precision == 'float32' or precision == 'autocast')
    return model

def load_img(path, size=None, tensor=True):
    image = Image.open(path).convert('RGB')
    w, h = image.size if size is None else size
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8
    image = image.resize((w,h), Image.Resampling.LANCZOS) # (w,h)
    if not tensor: return image
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(device)
    return 2.*image - 1., (w,h)

def save_img(image, num, out_dir, prefix='', filepath=None):
    image = torch.clamp((image + 1.) / 2., min=0., max=1.).permute(1,2,0).cpu().numpy() * 255
    if filepath is None: filepath = prefix + '%05d.jpg' % num
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(out_dir, filepath))

def makemask(img_path, mask, invert_mask=False, resize=True, threshold=0.35, model_path='models/clipseg/rd64-uni.pth'):
    if os.path.isfile(mask): 
        mask = load_img(mask, tensor=False)
        mask = mask.convert('L')
        mask = ImageOps.invert(mask)
    else: 
        image = load_img(img_path, tensor=False)
        txt2mask = Txt2Mask(model_path, device='cuda')
        mask = txt2mask.segment(image, mask).to_mask(float(threshold))
        # image.putalpha(mask)
    if invert_mask: mask = ImageOps.invert(mask)
    w, h = mask.size
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8
    if resize is True:
        mask = mask.resize((w//8, h//8), resample=Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4,1,1))[None] if resize is True else mask[None][None]
    mask = torch.from_numpy(mask).to(device)
    return mask

def prep_midas(image, model_type="dpt_hybrid"):
    transform = load_midas_transform(model_type)
    x = ((image + 1.) * .5).detach().permute(0,2,3,1).cpu().numpy()[0] # [h,w,c]
    x = transform({"image": x})["image"]
    x = torch.from_numpy(x[None, ...]).to(device)
    return x

def txt_clean(txt):
    if isinstance(txt, list):
        if isinstance(txt[0], tuple): txt = '. '.join([t[0] for t in txt]) # multi prompt with tuples
        else: txt = '. '.join(txt) # multi prompt 
    return ''.join(e for e in txt.replace(' ', '_') if (e.isalnum() or e in ['_','-']))

def read_txt(txt):
    if os.path.isfile(txt):
        with open(txt, 'r', encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        lines = [txt]
    return lines

def parse_line(txt):
    subs = []
    for subtxt in txt.split('|'):
        if ':' in subtxt:
            [subtxt, wt] = subtxt.split(':')
            wt = float(wt)
        else: wt = 1e-4 if len(subtxt.strip())==0 else 1.
        subs.append((subtxt.strip(), wt))
    return subs # list of tuples

def read_multitext(in_txt, prefix=None, postfix=None, flat=False):
    if in_txt is None or len(in_txt)==0: return []
    lines = read_txt(in_txt)
    if prefix  is not None: prefix  = read_txt(prefix)
    if postfix is not None: postfix = read_txt(postfix)
    if flat:
        prompts = []
        for i, tt in enumerate(lines):
            tt = tt.strip()
            if len(tt) == 0:
                prompts.append('')
            elif tt[0] != '#':
                if prefix  is not None: tt = prefix[i % len(prefix)] + ' | ' + tt
                if postfix is not None: tt = tt + ' | ' + postfix[i % len(postfix)]
                prompts.append(tt)
        weights = [1.] * len(prompts)
    else:
        prompts = [parse_line(tt) for tt in lines if tt.strip()[0] != '#']
        maxlen = 0
        for prompt in prompts:
            maxlen = max(maxlen, len(prompt))
        for i in range(len(prompts)):
            if len(prompts[i]) < maxlen:
                prompts[i] += [('', 1e-4)] * (maxlen - len(prompts[i]))
        if prefix is not None:
            prompts = [parse_line(prefix[i % len(prefix)]) + prompts[i] for i in range(len(prompts))]
        if postfix is not None:
            prompts = [prompts[i] + parse_line(postfix[i % len(postfix)]) for i in range(len(prompts))]
        weights = [[p[1] for p in prompt] for prompt in prompts]
        prompts = [[p[0] for p in prompt] for prompt in prompts]
    return prompts, weights # two lists if flat, or two lists of lists if multi

def unique_prefix(out_dir):
    dirlist = sorted(os.listdir(out_dir), reverse=True) # sort reverse alphabetically until we find max+1
    existing_name = next((f for f in dirlist if re.match('^(\d+)\..*\.jpg', f)), '00000.0.jpg')
    basecount = int(existing_name.split('.', 1)[0]) + 1
    return f'{basecount:05}'

def correct_colors(image_list, reference_image_path, image_callback = None): 
    reference_image = Image.open(reference_image_path)
    correction_target = cv2.cvtColor(np.asarray(reference_image), cv2.COLOR_RGB2LAB)
    for r in image_list:
        image, seed = r
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
        image = skimage.exposure.match_histograms(image, correction_target, channel_axis=2)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_LAB2RGB).astype("uint8"))
        if image_callback is not None:
            image_callback(image, seed)
        else:
            r[0] = image

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def file_list(path, ext=None):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if ext is not None: 
        files = [f for f in files if f.endswith(ext)]
    return sorted([f for f in files if os.path.isfile(f)])

def img_list(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['jpg', 'jpeg', 'png', 'ppm', 'tif']]
    return sorted([f for f in files if os.path.isfile(f)])


def blend(t, ip):
    if ip == 'bezier':
        return BezierBlend(t)
    elif ip == 'parametric':
        return ParametricBlend(t)
    elif ip == 'custom':
        return CustomBlend(t)
    else:
        return t

def BezierBlend(t):
  return t * t * (3.-2.*t)
def ParametricBlend(t):
    sqt = t*t
    return (sqt / (2.0 * (sqt - t) + 1.0))
def CustomBlend(x):
    return 1 - 2*x*(1-x) if x >= 0.5 else 2*x*(1-x)

def log_tokens(text, model):
    tokenized = ""
    discarded = ""
    usedTokens = 0
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', '')
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"{token} "
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"{token}"
    print(f".. {usedTokens} tokens: {tokenized}")
    if discarded != "": 
        print(f".. {totalTokens-usedTokens} discarded: {discarded}")

def read_latents(latents):
    if latents is not None and os.path.isfile(latents):
        key_latents = load_latents(latents)
    elif latents is not None and os.path.isdir(latents):
        key_latents = []
        lat_list = file_list(latents, 'pkl')
        for lat in lat_list: 
            key_latent = load_latents(lat)
            key_latents.append(key_latent)
        if isinstance(key_latents[0], list):
            key_latents = list(map(list, zip(*key_latents)))
            for n in range(len(key_latents)):
                key_latents[n] = torch.cat(key_latents[n])
        else:
            key_latents = torch.cat(key_latents)
    else:
        print(' No input latents found'); exit()
    return key_latents
    
def load_latents(lat_file):
    with open(lat_file, 'rb') as f:
        key_lats = pickle.load(f)
    idx_file = os.path.splitext(lat_file)[0] + '.txt'
    if os.path.exists(idx_file): 
        with open(idx_file) as f:
            lat_idx = f.readline()
            lat_idx = [int(l.strip()) for l in lat_idx.split(',') if '\n' not in l and len(l.strip())>0]
        key_lats = list(key_lats) if isinstance(key_lats, list) or isinstance(key_lats, tuple) else [key_lats]
        for n, key_lat in enumerate(key_lats):
            key_lats[n] = torch.cat([key_lat[i].unsqueeze(0) for i in lat_idx])
    return key_lats

# = = = progress bar = = = 

from shutil import get_terminal_size
import time
import ipywidgets as ipy
import IPython

class ProgressIPy(object):
    def __init__(self, task_num=10):
        self.pbar = ipy.IntProgress(min=0, max=task_num, bar_style='') # (value=0, min=0, max=max, step=1, description=description, bar_style='')
        self.labl = ipy.Label()
        IPython.display.display(ipy.HBox([self.pbar, self.labl]))
        self.task_num = task_num
        self.completed = 0
        self.start()

    def start(self, task_num=None):
        if task_num is not None:
            self.task_num = task_num
        if self.task_num > 0:
            self.labl.value = '0/{}'.format(self.task_num)
        else:
            self.labl.value = 'completed: 0, elapsed: 0s'
        self.start_time = time.time()

    def upd(self, *p, **kw):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin = ' end %s' % finaltime[11:16]
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            self.labl.value = '{}/{}, rate {:.3g}s, time {}s, left {}s, {}'.format(self.completed, self.task_num, 1./fps, shortime(elapsed), shortime(eta), fin)
        else:
            self.labl.value = 'completed {}, time {}s, {:.1f} steps/s'.format(self.completed, int(elapsed + 0.5), fps)
        self.pbar.value += 1
        if self.completed == self.task_num: self.pbar.bar_style = 'success'
        return self.completed

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''
    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal is small ({}), make it bigger for proper visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self, task_num=None):
        if task_num is not None:
            self.task_num = task_num
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def upd(self, msg=None, uprows=0):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin_msg = ' %ss left, end %s' % (shortime(eta), finaltime[11:16])
            if msg is not None: fin_msg += '  ' + str(msg)
            mark_width = int(self.bar_width * percentage)
            bar_chars = 'X' * mark_width + '-' * (self.bar_width - mark_width) # - - -
            sys.stdout.write('\033[%dA' % (uprows+2)) # cursor up 2 lines + extra if needed
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            try:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed, self.task_num, 1./fps, shortime(elapsed), shortime(eta), fin_msg))
            except:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed, self.task_num, 1./fps, shortime(elapsed), shortime(eta), '<< unprintable >>'))
        else:
            sys.stdout.write('completed {}, time {}s, {:.1f} steps/s'.format(self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

    def reset(self, count=None, newline=False):
        self.start_time = time.time()
        if count is not None:
            self.task_num = count
        if newline is True:
            sys.stdout.write('\n\n')

try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    progbar = ProgressIPy
except: # normal console
    progbar = ProgressBar

def time_days(sec):
    return '%dd %d:%02d:%02d' % (sec/86400, (sec/3600)%24, (sec/60)%60, sec%60)
def time_hrs(sec):
    return '%d:%02d:%02d' % (sec/3600, (sec/60)%60, sec%60)
def shortime(sec):
    if sec < 60:
        time_short = '%d' % (sec)
    elif sec < 3600:
        time_short  = '%d:%02d' % ((sec/60)%60, sec%60)
    elif sec < 86400:
        time_short  = time_hrs(sec)
    else:
        time_short = time_days(sec)
    return time_short


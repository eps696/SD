import os, sys
import time
import pickle
import argparse

import torch
from pytorch_lightning import seed_everything

from _sdrun import models, sd_setup
from utils import load_img, save_img, slerp, lerps, lerp, blend, calc_size, read_multitext, makemask

from utils import read_latents, img_list, basename, progbar

samplers = ['ddim', 'klms', 'euler', 'dpm_ada', 'dpm_fast']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-il', '--in_lats', default=None, help='Directory or file with saved keypoints to interpolate between')
    parser.add_argument('-ol', '--out_lats', default=None, help='File to save keypoints for further interpolation')
    parser.add_argument('-fs', '--fstep',   default=25, type=int, help="number of frames for each interpolation step")
    parser.add_argument(       '--curve',   default='linear', help="Interpolating curve: bezier, parametric, custom or linear")
    parser.add_argument(       '--loop',    action='store_true', help='Loop inputs [or keep the last one]')
    # inputs & paths
    parser.add_argument('-t',  '--in_txt',  default=None, help='Text string or file to process')
    parser.add_argument('-pre', '--prefix', default=None, help='Prefix for input text')
    parser.add_argument('-post', '--postfix', default=None, help='Postfix for input text')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images (overrides width and height)')
    parser.add_argument('-M',  '--mask',    default=None, help='Path to input mask for inpainting mode (overrides width and height)')
    parser.add_argument('-em', '--embeds',  default='_in/embed', help='Path to directories with embeddings')
    parser.add_argument('-o',  '--out_dir', default="_out", help="Output directory for generated images")
    parser.add_argument('-md', '--maindir', default='./', help='Main SD directory')
    # mandatory params
    parser.add_argument('-m',  '--model',   default='15', choices=models.keys(), help="model version [15,v21,v21v]")
    parser.add_argument('-sm', '--sampler', default='euler', choices=samplers)
    parser.add_argument('-nz', '--noiser',  default='default', choices=['default', 'karras', 'exponential', 'vp'])
    parser.add_argument(       '--vae',     default='ema', help='orig, ema, mse')
    parser.add_argument('-C','--cfg_scale', default=7.5, type=float, help="prompt guidance scale")
    parser.add_argument('-f', '--strength', default=0.75, type=float, help="strength of image processing. 0 = preserve img, 1 = replace it completely")
    parser.add_argument(      '--ddim_eta', default=0., type=float)
    parser.add_argument('-s', '--steps',    default=50, type=int, help="number of diffusion steps")
    parser.add_argument(     '--precision', default='autocast')
    parser.add_argument('-S', '--seed',     type=int, help="image seed")
    # custom diff with token mods
    parser.add_argument('-tm', "--token_mod", default=None, help="custom modifier token(s) to use with prompts")
    parser.add_argument('-d', "--delta_ckpt", default=None, help="path to delta checkpoint of fine-tuned custom diffusion block")
    parser.add_argument("--compress", action='store_true', help="if delta checkpoint is compressed")
    # misc
    parser.add_argument('-sz', '--size',    default=None, help="image sizes, multiple of 32")
    parser.add_argument('-inv', '--invert_mask', action='store_true')
    parser.add_argument('-v', '--verbose',  action='store_true')
    return parser.parse_args()

def main():
    # main setup
    a = get_args()
    [a, model, uc, img_lat, lat_z, lat_z_enc, img_z, rnd_z, txt_c, generate] = sd_setup(a)

    seed_everything(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    size = None if a.size is None else calc_size(a.size, a.model, a.verbose) 
    if model.uses_rml_inpainting or hasattr(model, 'depth_model'): print('!! depth/inpaint models cannot be used for interpolation !!'); exit()
    mask = lat_ = None
    if a.verbose: print('..', basename(models[a.model][1]), '..', a.sampler, '..', a.strength)
    
    # only images, no text prompts = interpolate & exit
    if a.in_img is not None and os.path.isdir(a.in_img):
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        count = len(img_paths)
        zs = []
        for i in range(count):
            init_image, (W,H) = load_img(img_paths[i], size)
            zs += [img_z(init_image)]
        pcount = count if a.loop else count-1
        pbar = progbar(pcount * a.fstep)
        for i in range(pcount):
            for j in range(a.fstep):
                z_ = slerp(zs[i], zs[i+1], j / a.fstep)
                image = generate(z_, uc, thresh=False) # thresholding fails if c=uc
                save_img(image, i*a.fstep+j, a.out_dir)
                pbar.upd(uprows=1)
        if not a.loop:
            image = generate(zs[pcount % count], uc, thresh=False)
            save_img(image, pcount*a.fstep, a.out_dir)
        exit()

    # single image + text prompts
    elif a.in_img is not None and os.path.isfile(a.in_img):
        init_image, (W,H) = load_img(a.in_img, size)
        if a.mask is not None:
            mask = makemask(a.in_img, a.mask, a.invert_mask)
            lat_ = img_lat(init_image)
        z_ = img_z(init_image)
    # only text prompts
    else:
        W, H = [models[a.model][2]]*2 if size is None else size
        z_ = None

    # load saved keypoints, if any
    if a.in_lats is not None and os.path.exists(a.in_lats):
        cs, zs, cws = read_latents(a.in_lats)
        cs, zs = cs.unsqueeze(1), zs.unsqueeze(1)
        count = len(cs)

    # prepare keypoints
    else:
        prompts, cws = read_multitext(a.in_txt, a.prefix, a.postfix) # cws = weights for prompts
        count = len(prompts)
        
        if a.out_lats is not None:
            lat_dir = 'lats/'
            os.makedirs(os.path.join(a.out_dir, lat_dir), exist_ok=True)
            pbar = progbar(count)

        cs = [] 
        zs = [] if z_ is None else [z_]
        for i, prompt in enumerate(prompts):
            cs += [txt_c(prompt)] # [1,77,768] condition
            if z_ is None: zs += [rnd_z(H//8, W//8)] # [1,4,64,64] noise

            if a.out_lats is not None:
                image = generate(zs[-1], cs[-1], cw=cws[i])
                save_img(image, i, a.out_dir, prefix=lat_dir)
                with open(a.out_lats, 'wb') as f:
                    pickle.dump((torch.cat(cs), torch.cat(zs), torch.tensor(cws[:(i+1)])), f)
                pbar.upd(uprows=1)

        if a.out_lats is not None: # save keypoints => exit
            print('zs', torch.cat(zs).shape, 'cs', torch.cat(cs).shape, 'cw', torch.tensor(cws).shape)
            exit()
        
    # interpolate
    pcount = count if a.loop else count-1
    pbar = progbar(pcount * a.fstep)
    for i in range(pcount):

        for f in range(a.fstep):
            tt = blend(f / a.fstep, a.curve)
            z_ = slerp(zs[i % len(zs)], zs[(i+1) % len(zs)], tt)
            c_ = slerp(cs[i % len(cs)], cs[(i+1) % len(cs)], tt)
            lerp_fn = lerps if isinstance(cws[0], list) else lerp
            cw = lerp_fn(cws[i], cws[(i+1) % len(cws)], tt)

            image = generate(z_, c_, cw=cw, img=lat_, mask=mask)
            save_img(image, i * a.fstep + f, a.out_dir)
            pbar.upd(uprows=1)

    if a.loop is not True:
        image = generate(zs[pcount % len(zs)], cs[pcount % len(cs)], cw=cws[pcount % len(cws)], img=lat_, mask=mask)
        save_img(image, pcount * a.fstep, a.out_dir)


if __name__ == '__main__':
    main()

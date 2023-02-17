# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.
import os
import argparse
import glob
import torch

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('-i', '--path', help='path of folder to checkpoints', type=str)
    parser.add_argument('-n', '--newtoken', help='number of new tokens in the checkpoint', default=1, type=int)
    return parser.parse_args()

def save_delta(ckpt_dir, newtoken=1):
    assert newtoken > 0, 'No new tokens found'
    layers = []
    for ckptfile in glob.glob(f'{ckpt_dir}/*.ckpt', recursive=True):
        if 'delta' not in ckptfile:
            st = torch.load(ckptfile)["state_dict"]
            if len(layers) == 0:
                for key in list(st.keys()):
                    if 'attn2.to_k' in key or 'attn2.to_v' in key:
                        layers.append(key)
            st_delta = {'state_dict': {}}
            for each in layers:
                st_delta['state_dict'][each] = st[each].clone()
            num_tokens = st['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'].shape[0]
            st_delta['state_dict']['embed'] = st['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'][-newtoken:].clone()
            filepath = os.path.join(ckpt_dir, 'delta-' + os.path.basename(ckptfile))
            torch.save(st_delta, filepath)
            os.remove(ckptfile)
            print('.. saved embedding', st_delta['state_dict']['embed'].cpu().numpy().shape, num_tokens, '=>', filepath)


if __name__ == "__main__":
    args = parse_args()
    save_delta(args.path, args.newtoken)
    
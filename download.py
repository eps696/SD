import os
from tqdm import tqdm
import urllib.request

def download_model(url: str, root: str = "./models"):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url.split('?')[0])
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=64, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    return download_target

print(' downloading SD 1.5 model')
download_model("https://www.dropbox.com/s/k9odmzadgyo9gdl/sd-v15-512-fp16.ckpt?dl=1", 'models')
print(' downloading SD 1.5-inpainting model')
download_model("https://www.dropbox.com/s/cc5usmoik43alcc/sd-v15-512-inpaint-fp16.ckpt?dl=1", 'models')
print(' downloading SD 2-inpainting model')
download_model("https://www.dropbox.com/s/kn9jhrkofsfqsae/sd-v2-512-inpaint-fp16.ckpt?dl=1", 'models')
print(' downloading SD 2-depth model')
download_model("https://www.dropbox.com/s/zrx5qfesb9jstsg/sd-v2-512-depth-fp16.ckpt?dl=1", 'models')
print(' downloading SD 2.1 model')
download_model("https://www.dropbox.com/s/m4v36h8tksqa2lk/sd-v21-512-fp16.ckpt?dl=1", 'models')
print(' downloading SD 2.1-v model')
download_model("https://www.dropbox.com/s/wjzh3l1szauz5ww/sd-v21v-768-fp16.ckpt?dl=1", 'models')

print(' downloading OpenCLIP ViT-H-14-laion2B-s32B-b79K model')
download_model("https://www.dropbox.com/s/7smohfi2ijdy1qm/laion2b_s32b_b79k-vit-h14.pt?dl=1", 'models/openclip')
# download_model("https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin", 'models/openclip')

print(' downloading SD VAE-ema model')
download_model("https://www.dropbox.com/s/dv836z05lblkvkc/vae-ft-ema-560000.ckpt?dl=1", 'models')
print(' downloading SD VAE-mse model')
download_model("https://www.dropbox.com/s/jmxksbzyk9fls1y/vae-ft-mse-840000.ckpt?dl=1", 'models')

print(' downloading CLIPseg model')
download_model("https://www.dropbox.com/s/c0tduhr4g0al1cq/rd64-uni.pth?dl=1", 'models/clipseg')
print(' downloading MiDaS depth model')
download_model("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt", 'models/depth')


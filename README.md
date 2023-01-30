# Stable Diffusion for students

<p align='center'><img src='_in/something.jpg' /></p>

This is yet another Stable Diffusion compilation, aimed to be functional, clean & compact enough for various experiments. No GUI, of course.  
Functionality includes: txt2img, img2img, depth/inpainting models, embeddings, weighted multi-prompts, latent interpolations, to be continued..  
More details, Colab version and finetuning will follow soon. 

The code is based on CompVis library and heavily borrows from [this repo](https://github.com/AmericanPresidentJimmyCarter/stable-diffusion), with occasional additions from [InvokeAI](https://github.com/invoke-ai/InvokeAI) and [Deforum](https://github.com/deforum/stable-diffusion).  

## Setup

Install CUDA 11.6. If you're on Windows, you'll need Visual Studio 2019 for `xformers` library (it significantly enhancing performance).  
Setup environment:
```
conda create -n SD python=3.10 numpy pillow 
activate SD
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
rem conda install pytorch=1.12 torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/xformers.git
```
Download models: Stable Diffusion (1.5, 1.5-inpaint, 2-inpaint, 2-depth, 2.1, 2.1-v), OpenCLIP ViT-H-14, CLIPseg, MiDaS. Please visit their webpages for licensing info. Most model files are converted to `float16` for faster loading. 
```
python download.py
```

## Operations

Examples of usage (as raw command lines or using Windows bat-files):

* Generate an image from the text prompt:
```
python libs/_sdrun.py -t "hello world" --size 1024-576
```
* Redraw an image with existing style embedding, keeping the depth map intact:
```
python libs/_sdrun.py -im _in/something3d.jpg -t "<art-brut>" --model v2d
```
* Inpaint directory of images with RunwayML model, turning humans into animals:
```
python libs/_sdrun.py -im _in/pix --mask "humans, people" -t "animals, beasts" --model 15i
```
* Make a video, interpolating between the lines of the text file:
```
python libs/latwalk.py -t yourfile.txt --fstep 50
```
* The same, drawing over a masked image:
```
python libs/latwalk.py -t yourfile.txt -im myphoto.jpg --mask mymask.jpg
```
Text prompts may include embeddings, e.g. `<depthmap>`, and weights, like `good prompt :1 | also good prompt :1 | bad prompt :-0.5`.

## Credits

Huge respect to the people behind [Stable Diffusion], [InvokeAI], [Deforum], 

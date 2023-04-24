# import torch
# from diffusers import StableDiffusionImageVariationPipeline, StableDiffusionPipeline
# from PIL import Image
# from torchvision import transforms
#
# # pip install diffusers==0.11.1
#
# device = "cpu"
# sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
#   "lambdalabs/sd-image-variations-diffusers",
#   revision="v2.0",
#   )
# sd_pipe = sd_pipe.to(device)
#
# im = Image.open("/home/cvpr/Desktop/bird.jpg")
# tform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(
#         (224, 224),
#         interpolation=transforms.InterpolationMode.BICUBIC,
#         antialias=False,
#     ),
#     transforms.Normalize(
#         [0.48145466, 0.4578275, 0.40821073],
#         [0.26862954, 0.26130258, 0.27577711]),
# ])
# inp = tform(im).to(device).unsqueeze(0)
#
# out = sd_pipe(inp, guidance_scale=3)
# out["images"][0].save("result.jpg")

import os
from pathlib import Path
from lambda_diffusers import StableDiffusionImageEmbedPipeline
import glob
from PIL import Image
import torch
import torch.nn as nn

path = '/home2/coremax/Documents/imagenet-mini/train/'
num_samples = 3
base_path = Path("outputs")

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device = "cpu"
pipe = StableDiffusionImageEmbedPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers",
                                                         revision="273115e88df42350019ef4d628265b8c29ef4af5")
pipe = pipe.to(device)


for img in sorted(glob.glob(path + '*/*.*')):
    print(img)
    im = Image.open(img)

    image = pipe(num_samples * [im], guidance_scale=3.0, num_inference_steps=15)
    image = image["sample"]

    for idx, im in enumerate(image):
        img_name = img.rsplit('/', 1)[1]
        im.save(base_path / str(str(idx) + img_name))

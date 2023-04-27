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


from pathlib import Path
from lambda_diffusers import StableDiffusionImageEmbedPipeline
import glob
from PIL import Image
import torch

path = '/home2/coremax/Documents/imagenet-mini/train/'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"
pipe = StableDiffusionImageEmbedPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", revision="273115e88df42350019ef4d628265b8c29ef4af5")
pipe = pipe.to(device)

num_samples = 5

for img in glob.glob(path + '*/*.*'):
    im = Image.open(img)

    image = pipe(num_samples * [im], guidance_scale=3.0)
    image = image["sample"]

    base_path = Path("outputs/im2im")
    base_path.mkdir(exist_ok=True, parents=True)

    for idx, im in enumerate(image):
        im.save(base_path / f"{idx:06}.jpg")




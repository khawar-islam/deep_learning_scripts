# import the necessary packages
from matplotlib import pyplot as plt
from easyocr import Reader
import argparse
import cv2
from PIL import ImageFont, ImageDraw, Image

# import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

font = ImageFont.truetype("gulim.ttc", 25)

#print(cv2.__version__)


# conda create -n timm_tutorials python=3.7 (installed Python 3.7.16)
# conda activate timm_tutorials
# pip3 install -r requirments.txt
# pip3 install matplotlib == 3.3.1
# pip3 install opencv-python==4.4.0.42

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


args = {
    "image": "/media/cvpr/CM_24/03.jpg",
    "langs": "ko,en",
    "gpu": 1
}

# break the input languages into a comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))

# load the input image from disk
image = cv2.imread(args["image"])
cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

pil_im = Image.fromarray(cv2_im_rgb)
draw = ImageDraw.Draw(pil_im)
# korean_g2,
reader = Reader(langs, recog_network='korean_g2', gpu=args["gpu"] > 0)
print(reader.character)
results = reader.readtext(image)

b, g, r, a = 139, 0, 0, 0
# loop over the results
for (bbox, text, prob) in results:
    # display the OCR'd text and associated probability
    print("[INFO] {:.4f}: {}".format(prob, text))

    # unpack the bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    # img = cv2.rectangle(np.array(pil_im), tl, br, (0, 255, 0), 2)

    draw.text((tl[0], tl[1] - 10), text, font=font, fill=255)
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imwrite("resultb.png", cv2_im_processed)

# show the output image
cv2.imshow("Image", cv2_im_processed)
cv2.waitKey(0)

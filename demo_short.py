import os
os.environ['USE_TORCH'] = '1'
import torch
from collections import OrderedDict
from doctr.datasets import VOCABS
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, master
import matplotlib.image as img
import matplotlib
import matplotlib.font_manager
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import rc
from doctr.utils.visualization import visualize_page, draw_boxes
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
from doctr.io.image import read_img_as_tensor
rc('font', family='NanumGothic')

doc = DocumentFile.from_images("/media/cvpr/CM_24/KR_2M/val/images/개발하는 연구 분야이다._015.png")
model = master(pretrained=False, vocab=OrderedDict.fromkeys(VOCABS['korean']))
model.load_state_dict(torch.load("references/recognition/master_20230324-095826.pt", map_location="cpu"))
predictor = ocr_predictor(reco_arch=model)
model = predictor.reco_predictor(doc)
print(model)
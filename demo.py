import os

os.environ['USE_TORCH'] = '1'
from doctr.models import master, db_resnet50, crnn_vgg16_bn, vitstr_base
import doctr
import glob
import torch
import cv2
import os
from doctr.models import ocr_predictor
from skimage.io import imread_collection
from doctr.datasets.vocabs import VOCABS
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import numpy as np
from doctr.utils.visualization import visualize_page, draw_boxes
from collections import OrderedDict
import matplotlib.pyplot as mpl
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname='/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf').get_name()
rc('font', family=font_name)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from matplotlib import font_manager as fm
from hocr import HocrParser

'''
--path /media/cvpr/CM_1/COREMAX/extra/aaa.jpg --recognition /media/cvpr/CM_2/doctr/master_20220916-173609.pt
'''

# Enable GPU growth if using TF
if is_tf_available():
    import tensorflow as tf

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if any(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

CLASSES = ["__background__", "QR Code", "Barcode", "Logo", "Photo"]
CM = [(255, 255, 255), (0, 0, 150), (0, 0, 0), (0, 150, 0), (150, 0, 0)]


def main(args):
    REC_CKPT = "references/recognition/master_20230324-095826.pt"  # or
    model = master(pretrained=False, vocab=OrderedDict.fromkeys(VOCABS['korean']))
    #model = master(pretrained=False, vocab=OrderedDict.fromkeys(VOCABS['korean'] + VOCABS['english']))

    model.load_state_dict(torch.load(REC_CKPT, map_location='cpu'), strict=False)
    model = ocr_predictor(reco_arch=model, pretrained=True).cuda()

    if args.path.endswith(".pdf"):
        doc = DocumentFile.from_pdf(args.path)
    else:
        doc = DocumentFile.from_images(args.path)

    out = model(doc)

    out.show(doc)

    # synthetic_pages = out.synthesize()
    # img = plt.imshow(synthetic_pages[0], interpolation='nearest')
    # img.set_cmap('hot')
    # plt.axis('off')
    # plt.savefig('reconstruction.jpg', dpi=1000, bbox_inches='tight')
    # plt.show()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path', type=str, help='Path to the input document (PDF or image)',
                        default='/home/cvpr/Desktop/fdsfsd.png')
    parser.add_argument('--detection', type=str, default='db_resnet50',
                        help='Text detection model to use for analysis')
    parser.add_argument('--recognition', type=str, default='',
                        help='Text recognition model to use for analysis')
    parser.add_argument("--noblock", dest="noblock", help="Disables blocking visualization. Used only for CI.",
                        action="store_true")
    parser.add_argument("--static", dest="static", help="Switches to static visualization", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

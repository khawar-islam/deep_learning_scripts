import json
import os
import re
import glob

# Input (number of images i1,i2,i3)
# Output (labels.json file contains {"abc_1.jpg": "abc"})


path = "/media/cvpr/CM_1/ocr_kor/data/generator/TextRecognitionDataGenerator/out/images/"
save_path = ""

lab_key = []
lab_string = []

lab_dic = {}

for img in glob.glob(path + '*.*'):
    #print(os.path.basename(img))
    key = os.path.basename(img)
    lab_key = os.path.basename(img)
    lab_string = key.split('_')[0]

    lab_dic[lab_key] = lab_string



with open("labels.json", "w", encoding="utf-8") as outfile:
    json.dump(lab_dic, outfile, ensure_ascii=False)



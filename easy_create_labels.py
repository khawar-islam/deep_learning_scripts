import pandas as pd
import json
import os

# # Read space-separated columns without header
# data = pd.read_csv('/media/cvpr/CM_24/synthtiger/results/gt.txt', sep="\s+", header=None)
# # Update columns
# data.columns = ['filename', 'words']
#
# data['filename'] = data['filename'].apply(os.path.basename)
#
# # Save to required format
# data.to_csv('/media/cvpr/CM_24/synthtiger/results/labels.csv', index=False)

#=======================================================================

f = open('/media/cvpr/CM_1/ocr_kor/data/generator/TextRecognitionDataGenerator/out/labels.json')

json = json.load(f)

df = pd.DataFrame({'filename': json.keys(), 'words': json.values()})
#df['filname'] = df['filname'].replace(r'.+_', '', regex=True)
df.to_csv('/media/cvpr/CM_1/ocr_kor/data/generator/TextRecognitionDataGenerator/out/labels.csv', index=False)

# Rename files

# import os
# import glob
#
# x = glob.glob("/media/cvpr/CM_24/KR_2M/train/images/*.*")
# current_dir = "/media/cvpr/CM_24/KR_2M/train/images/"
#
# for img in x:
#     key = os.path.basename(img)
#     lab_key = os.path.basename(img)
#     lab_string = key.split('_')[1]
#     os.rename(img, os.path.join(current_dir, lab_string))

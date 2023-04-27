import os
import shutil

images = os.listdir(os.getcwd())


ids = set(i.split('_')[0] for i in images)  # set removes duplicates
ids.remove("morph.py")
print(ids)
for i in ids:
    os.mkdir(i)  # create subdirs
for img in images:
    target_dir = img.split('_')[0]
    shutil.move(img, target_dir)

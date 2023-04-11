import shutil
import os
import glob
import natsort

mapping = {}
f = open('/media/cvpr/CM_24/synthtiger/results/gt.txt', 'r')

for line in f.readlines():
    k, v = line.split()
    path, old_name = k.rsplit('/', 1)
    new_name = "{}/{}_{}".format(path, v, old_name)
    mapping[k] = new_name.rsplit('/', 1)[1]

 # new_name.rsplit('/', 1)[1]
print(mapping[k])

base_path = '/media/cvpr/CM_24/synthtiger/results/'
base_path1 = '/media/cvpr/CM_24/KR_2M/train/images'
for old, new in mapping.items():
    old_path = os.path.join(base_path, old)
    new_path = os.path.join(base_path1, new)
    os.rename(old_path, new_path)


# path = '/media/cvpr/CM_22/synthtiger/results/images/'
# with open('/media/cvpr/CM_22/synthtiger/results/gt.txt') as f:
#     lines = f.readlines()
#
# for root, dirs, files in sorted(os.walk(path)):
#     for file in natsort.natsorted(files):
#         for line in lines:
#             if line.find(file):
#                 print(file)
#                 print('Line Number:', lines.index(line))
#                 print('Line:', line)
#
#                 korean_label = line.split()
#                 print(korean_label[1])
#                 print(korean_label[1] + "_" + file)

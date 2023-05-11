import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision

class GridMask(object):

    '''Perform GridMask Data Augmentation.
    Args:
       num_grids : number of grids If -1 is given it will randomly
                   select n between (2, 5).
       grid_size : size of the each removed region. If -1 is given it will
                   randomly select a squared region between [Image_shape//8, Image_shape//4]
    '''
    def __init__(self,
                 num_grids = -1,
                 grid_size = -1,
                 # rotate = 2
                 ):
        super(GridMask, self).__init__()
        self.num_grids = num_grids
        self.grid_size = grid_size
        # self.rotate = rotate

    def __call__(self, img):
        num_grids = self.num_grids if self.num_grids > 0 else np.random.randint(2, 5)
        grid_size = self.grid_size if self.grid_size > 0 else np.random.randint(img.size()[-1 ] /8, img.size()[-1 ] /4)

        # print(num_grids, grid_size)
        mask = np.ones((img.size()[-1], img.size()[-1]), np.float32)

        # Choose starting point.
        start_x = np.random.randint(0, (img.size()[-1 ] -grid_size )//num_grids)
        start_y = np.random.randint(0, (img.size()[-1 ] -grid_size )//num_grids)
        temp = start_x

        for i in range(num_grids):
            start_x = temp
            for j in range(num_grids):
                mask[start_y : start_y + grid_size, start_x : start_x + grid_size] = 0.
                start_x += 2* grid_size
            start_y += 2 * grid_size

        # r = np.random.randint(self.rotate)
        # mask = Image.fromarray(np.uint8(mask))
        # mask = mask.rotate(r)
        # mask = np.asarray(mask)

        mask = torch.from_numpy(mask).float()

        mask = mask.expand_as(img)
        img = img * mask

        return img


# Preprocess
PREPROCESS = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
    ]
)

pil_img = Image.open('/media/cvpr/CM_24/atu/images/atu/atu.png')


tensor_img = PREPROCESS(pil_img)

gridmask = GridMask()
gridmasked_x = gridmask(tensor_img)

# Convert the Tensor back to Image for visualization Purpose.
convert_img = transforms.ToPILImage()

#f, ax = plt.subplots(1, 1)

#ax.imshow(convert_img(gridmasked_x))

img = plt.imshow(convert_img(gridmasked_x), interpolation='nearest')
img.set_cmap('hot')
plt.axis('off')
plt.savefig("test.png", bbox_inches='tight', dpi=200)

# plt.savefig('base.png', format='png')
# plt.minorticks_off()
# plt.show()
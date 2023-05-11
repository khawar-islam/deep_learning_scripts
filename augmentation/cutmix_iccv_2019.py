import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets
from cutmix.cutmix.cutmix import CutMix

train_dataset = torchvision.transforms.Compose([
    transforms.Resize((500, 400)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.ToTensor()])

training_data = datasets.ImageFolder(root="/media/cvpr/CM_24/atu", transform=train_dataset)
train_loader = CutMix(training_data, num_class=1, beta=1.0, prob=0.5, num_mix=2)

for i, (images, target) in enumerate(train_loader):
    save_image(images, 'cutmix' + str(i) + '.png')

import numpy as np

from torch.utils.data import Dataset
from torch import Tensor

def oneHotTransform(label):
    arr = np.zeros(10)
    arr[label] = 1
    return arr

class KMNISTImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, is_train=True, source_folder=""):
        tag = "train" if is_train else "test"
        self.arr_train_data = np.load(source_folder + f"kmnist-{tag}-imgs.npz")['arr_0']
        self.arr_train_labels = np.load(source_folder + f"kmnist-{tag}-labels.npz")['arr_0']
        self.transform = transform
        self.target_transform = oneHotTransform

    def __len__(self):
        return self.arr_train_labels.size

    def __getitem__(self, idx):
        image_org = self.arr_train_data[idx]
        image = np.zeros([3*28*28]).reshape([3,28,28])
        image[0,:,:] = image_org
        image[1,:,:] = image_org
        image[2,:,:] = image_org
        image = Tensor(image.astype(np.double))
        
        label = self.arr_train_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
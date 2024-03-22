import numpy as np
import os
import pickle
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, Flowers102
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import os
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))


class CIFARDataset(Dataset):
    def __init__(self, H, W, n=3, seed=2):
        np.random.seed(seed)

        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.CenterCrop((W, H))
        ])

        cifar = CIFAR10(dir_path, download=True)
        cifar_data = cifar.data[np.array(cifar.targets) == 5]

        choices = np.random.choice(np.arange(len(cifar_data)), n, replace=False)

        self.data = cifar_data[choices]
        self.channels = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = self.transform(self.data[idx]).unsqueeze(0)[0]
        return (raw - raw.min()) / (raw.max() - raw.min()), 5

    def save_image(self, idx, file_path):
        """
        Saves the image at the specified index to the given file path.

        Parameters:
        - idx (int): The index of the image in the dataset.
        - file_path (str): The path (including file name) where the image will be saved.
        """
        # Ensure the index is within bounds
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        # Get the image tensor
        image_tensor = self.__getitem__(idx)

        # Convert the tensor to a PIL Image
        image = Image.fromarray((image_tensor.numpy() * 255).astype(np.uint8).transpose(1, 2, 0))

        # Check and create directory if necessary
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the image
        image.save(file_path)

class FlowersDataset(Dataset):
    def __init__(self, H, W, n=3):
        np.random.seed(37)

        self.transform = v2.Compose([
            v2.ToImageTensor(),
            v2.Resize((W, H), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
        ])

        flowers = Flowers102(dir_path, transform=self.transform, download=True)

        choices = np.random.choice(np.arange(len(flowers)), n, replace=False)
        choices[-1] += 1
        self.data = Subset(flowers, choices)
        self.channels = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = self.data[idx][0].data
        return (raw - raw.min()) / (raw.max() - raw.min())


class CelebDataset(Dataset):
    def __init__(self, H, W, n=3):
        with open(os.path.join(dir_path, 'celeb_data.pkl'), 'br') as f:
            data = pickle.load(f)
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.Resize((W, H), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
        ])

        self.data = [self.transform(img) for img in data[:n]]
        self.channels = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = self.data[idx]
        return (raw - raw.min()) / (raw.max() - raw.min()), 0

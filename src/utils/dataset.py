import os
import os.path as osp
from pathlib import Path
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert osp.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = osp.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class UnalignedDataset(data.Dataset):
    def __init__(self, dataroot, image_size=224, is_train=True):
        self.dataroot = Path(dataroot)
        self.image_size = image_size
        self.image_size_big = int(image_size*1.2)
        self.dir_A = self.dataroot / "A"
        self.dir_B = self.dataroot / "B"
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.is_train = is_train
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def __len__(self):
        return min(len(self.A_paths), len(self.B_paths))

    def denormalize(self, imgs):
        denorm = transforms.Normalize(
            mean=(-0.5/0.5, -0.5/0.5, -0.5/0.5), std=(1/0.5, 1/0.5, 1/0.5))
        if imgs.dim() == 4:
            imgs = torch.stack([denorm(img) for img in imgs])
        elif imgs.dim() == 3:
            imgs = denorm(imgs)
        return imgs

    def __getitem__(self, idx):
        # Get A img and B img
        A_path = self.A_paths[idx % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        # Read imgs
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        if self.is_train:
            # Resize
            A_img = TF.resize(
                A_img, size=(self.image_size_big, self.image_size_big),
                interpolation=Image.ANTIALIAS)
            B_img = TF.resize(
                B_img, size=(self.image_size_big, self.image_size_big),
                interpolation=Image.ANTIALIAS)
            # RandomCrop
            i, j, h, w = transforms.RandomCrop.get_params(
                A_img, output_size=(self.image_size, self.image_size))
            A_img = TF.crop(A_img, i, j, h, w)
            B_img = TF.crop(B_img, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                A_img = TF.hflip(A_img)
                B_img = TF.hflip(B_img)
            # Random vertical flipping
            if random.random() > 0.5:
                A_img = TF.vflip(A_img)
                B_img = TF.vflip(B_img)
        # No augmentation for testing
        else:
            A_img = TF.resize(
                A_img, size=(self.image_size, self.image_size),
                interpolation=Image.ANTIALIAS)
            B_img = TF.resize(
                B_img, size=(self.image_size, self.image_size),
                interpolation=Image.ANTIALIAS)
        A_img = TF.to_tensor(A_img)
        B_img = TF.to_tensor(B_img)
        # 0~1 => -1~1
        A_img = TF.normalize(A_img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        B_img = TF.normalize(B_img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        return {"A": A_img, "B": B_img, "A_path": A_path, "B_path": B_path,
                "class_A": torch.FloatTensor([0]),
                "class_B": torch.FloatTensor([1])}

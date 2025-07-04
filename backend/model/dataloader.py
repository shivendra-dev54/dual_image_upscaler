import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


class DualInputSuperResDataset(Dataset):
    def __init__(self, hr_dir, scale_factor=4, patch_size=128):
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size

        self.hr_images = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Transforms
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hr_images)

    def create_dual_inputs(self, hr_image):
        """Create two different low-resolution versions of the same image"""
        lr_size = self.patch_size // self.scale_factor

        # Method 1: Different downscaling methods
        lr_image1 = hr_image.resize((lr_size, lr_size), Image.BICUBIC)
        lr_image2 = hr_image.resize((lr_size, lr_size), Image.LANCZOS)

        # Method 2: Add slight variations (noise, blur, etc.)
        # Convert to numpy for processing
        lr1_np = np.array(lr_image1)
        lr2_np = np.array(lr_image2)

        # Add slight gaussian noise to first image
        noise = np.random.normal(0, 2, lr1_np.shape)
        lr1_np = np.clip(lr1_np + noise, 0, 255).astype(np.uint8)

        # Add slight gaussian blur to second image
        lr2_np = cv2.GaussianBlur(lr2_np, (3, 3), 0.5)

        # Convert back to PIL
        lr_image1 = Image.fromarray(lr1_np)
        lr_image2 = Image.fromarray(lr2_np)

        return lr_image1, lr_image2

    def __getitem__(self, idx):
        # Load HR image
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        hr_image = Image.open(hr_path).convert('RGB')

        # Random crop for training
        w, h = hr_image.size
        if w < self.patch_size or h < self.patch_size:
            hr_image = hr_image.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.LANCZOS)
            w, h = hr_image.size

        # Random crop
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr_image = hr_image.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Create dual LR inputs
        lr_image1, lr_image2 = self.create_dual_inputs(hr_image)

        # Convert to tensors
        hr_tensor = self.hr_transform(hr_image)
        lr_tensor1 = self.lr_transform(lr_image1)
        lr_tensor2 = self.lr_transform(lr_image2)

        return lr_tensor1, lr_tensor2, hr_tensor

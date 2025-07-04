import os
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from model import DualInputDiscriminator, DualInputESRGANGenerator

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
generator = DualInputESRGANGenerator().to(device)
discriminator = DualInputDiscriminator().to(device)

# Load weights if available
gen_ckpt_path = "./models/dual_input_esrgan_generator.pth"
disc_ckpt_path = "./models/dual_input_esrgan_discriminator.pth"

if os.path.exists(gen_ckpt_path):
    generator.load_state_dict(torch.load(gen_ckpt_path, map_location=device))
    print(f"Loaded generator weights from: {gen_ckpt_path}")
else:
    print("Generator weights not found. Using random weights.")

if os.path.exists(disc_ckpt_path):
    discriminator.load_state_dict(torch.load(disc_ckpt_path, map_location=device))
    print(f"Loaded discriminator weights from: {disc_ckpt_path}")
else:
    print("Discriminator weights not found. Using random weights.")

def upscale_dual_images(image1_path, image2_path, output_path):
    """Upscale using two input images to create one enhanced output"""
    generator.eval()

    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    min_w, min_h = min(img1.size[0], img2.size[0]), min(img1.size[1], img2.size[1])
    img1 = img1.resize((min_w, min_h), Image.LANCZOS)
    img2 = img2.resize((min_w, min_h), Image.LANCZOS)

    transform = transforms.ToTensor()
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_img = generator(img1_tensor, img2_tensor)
        sr_img = sr_img.squeeze(0).cpu()
        sr_img = torch.clamp(sr_img, 0, 1)
        sr_img = transforms.ToPILImage()(sr_img)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sr_img.save(output_path)
    print(f"Upscaled image saved to: {output_path}")
    return sr_img

def create_comparison_image(image1_path, image2_path, output_path):
    """Create side-by-side comparison image: bicubic1 | bicubic2 | ESRGAN"""
    generator.eval()

    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    min_w, min_h = min(img1.size[0], img2.size[0]), min(img1.size[1], img2.size[1])
    img1 = img1.resize((min_w, min_h), Image.LANCZOS)
    img2 = img2.resize((min_w, min_h), Image.LANCZOS)

    transform = transforms.ToTensor()
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_img = generator(img1_tensor, img2_tensor)

        comparison = torch.cat([
            F.interpolate(img1_tensor, scale_factor=4, mode='bicubic', align_corners=False),
            F.interpolate(img2_tensor, scale_factor=4, mode='bicubic', align_corners=False),
            sr_img
        ], dim=3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vutils.save_image(comparison, output_path, normalize=True)
    print(f"Comparison image saved to: {output_path}")
    return comparison


if __name__ == "__main__":
    print("Dual-Input ESRGAN Inference")

    image1_path = input("Enter path to the first input image: ").strip()
    image2_path = input("Enter path to the second input image: ").strip()
    output_path = input("Enter path to save the upscaled output image (e.g. results/sr_output.png): ").strip()
    comparison_path = input("Enter path to save comparison image (or press Enter to skip): ").strip()

    upscale_dual_images(image1_path, image2_path, output_path)

    if comparison_path:
        create_comparison_image(image1_path, image2_path, comparison_path)

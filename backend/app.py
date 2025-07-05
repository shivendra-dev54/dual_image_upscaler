from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # <-- added

import torch
from model.model import DualInputESRGANGenerator
from torchvision import transforms
from PIL import Image
from io import BytesIO

app = FastAPI()

# ✅ Allow CORS (allow frontend to access this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
generator = DualInputESRGANGenerator().to(device)
generator.load_state_dict(torch.load("model/models/dual_input_esrgan_generator.pth", map_location=device))
generator.eval()

transform = transforms.ToTensor()

@app.post("/upscale/")
async def upscale(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = Image.open(BytesIO(await file1.read())).convert("RGB")
    img2 = Image.open(BytesIO(await file2.read())).convert("RGB")

    print("images loaded")

    # Resize to smallest common shape
    w, h = min(img1.width, img2.width), min(img1.height, img2.height)
    img1 = img1.resize((w, h))
    img2 = img2.resize((w, h))

    input1 = transform(img1).unsqueeze(0).to(device)
    input2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_img = generator(input1, input2)
        sr_img = sr_img.squeeze(0).cpu()
        sr_img = torch.clamp(sr_img, 0, 1)
        out_image = transforms.ToPILImage()(sr_img)
    
    print("image processing complete!!!")

    buf = BytesIO()
    out_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

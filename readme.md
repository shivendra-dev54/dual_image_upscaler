# Dual-Image Super Resolution for High-Resolution Optical Satellite Imagery

A project submission for **ISRO Antariksh Hackathon 2025**

Team of 4 | Problem Statement: **"Dual image super resolution for high-resolution optical satellite imagery and its Blind Evaluation"**

---

## 🌍 Overview

This project enhances the resolution of satellite images using a **dual-input super-resolution model** based on ESRGAN. It takes **two temporally or spatially nearby low-resolution images** as input and reconstructs a **single high-resolution image**. It is designed for scenarios where only limited-quality satellite imagery is available and accurate upscaling is critical for earth observation, planning, and analysis.

The solution consists of:

* A custom **PyTorch-based dual-input ESRGAN model**
* A **training pipeline** using the DIV2K dataset
* A user-friendly **frontend interface**
* A **backend API service** for inference

---

## 🚀 Features

* Dual-input ESRGAN architecture with perceptual and consistency losses
* Blind evaluation setup for model robustness
* Flexible training with patch-based dataset loading
* Comparison image generation for qualitative evaluation
* Inference support via script and web API (optional)

---

## 💡 Use Case

This model is targeted toward satellite image enhancement using **multi-temporal** or **multi-angle** observations, such as:

* Vegetation and land use monitoring
* Urban mapping and infrastructure analysis
* Change detection in disaster-affected zones

---

## 🚧 Project Structure

```
project_dual_image_upscaler/
├── backend/
│   ├── model/
│   │   ├── inference.py            # Inference script (CLI)
│   │   ├── train.py                # Model training script
│   │   ├── model.py                # ESRGAN generator/discriminator
│   │   ├── utils.py                # Loss functions and helpers
│   │   ├── dataloader.py          # Dual-input dataset
│   │   ├── models/
│   │   │   ├── dual_input_esrgan_generator.pth
│   │   │   └── dual_input_esrgan_discriminator.pth
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages
│   │   │   ├── MainPage.tsx
│   │   │   └── HomePage.tsx
│   │   ├── Components
│   │   │   └── Navbar.tsx
│   ├── package.json
│
├── data/
│   ├── train_hr/                 # High-resolution training images
│   └── train_lr/                 # Generated low-res training pairs
│
├── README.md                    # Project overview and usage
└── .gitignore
```

---

## 📆 Setup Instructions

### Model training

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Download Training Data

Use the following utility:

```bash
python model/data_setup.py
```

#### Train the Model

```bash
python model/train.py
```

#### Inference (CLI)

```bash
python model/inference.py
```

Follow the prompts to provide two input images and get the upscaled output.

---

### Backend
For starting server to use model with web page efficiently.
```bash
uvicorn app:app
```

---
### Frontend
In a different terminal tab.
```bash
cd frontend
npm i
npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## 🏆 Team Members

* Shivendra Devadhe *(Machine Learning)*
* Shrikant Adhav *(Frontend Dev)*
* Jeevika Agrawal *(Backend Dev)*
* Sanika Desai *(Research & Documentation)*

---

## 🔧 Technologies Used

* Python, PyTorch, TorchVision
* OpenCV, NumPy, Matplotlib
* FastAPI 
* React + Vite (Frontend)

---

## 📊 Results & Evaluation

* Quantitative Metrics: PSNR, SSIM (blind evaluation compatible)
* Visual inspection through side-by-side comparisons

---

## 📅 Submission for ISRO Antariksh Hackathon 2025

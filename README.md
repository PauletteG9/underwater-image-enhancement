# 🌊 Hybrid Underwater Image Enhancement using CNN

## 📌 Overview

This project presents a **hybrid deep learning pipeline** for underwater image enhancement, combining:

* **Blind Denoising (DnCNN-style)**
* **Residual CNN-based Color Enhancement**

Underwater images suffer from **color distortion, low contrast, and noise** due to light absorption and scattering. This model improves visual quality while preserving structural details.

---

## 🎯 Objectives

* Remove noise using blind denoising
* Restore color balance using perceptual color correction
* Enhance contrast and sharpness
* Preserve structural details
* Evaluate performance using multiple image quality metrics

---

## 🧠 Model Architecture

### 🔹 Stage 1: Blind Denoiser

* DnCNN-style architecture
* Learns **residual noise map**
* Output: Cleaned image

### 🔹 Stage 2: Color Enhancer

* Residual CNN with:

  * ResBlocks
  * Channel Attention (Squeeze-and-Excitation)
  * Global skip connection
* Output: Enhanced image with improved color & contrast

---

## ⚙️ Training Details

* **Framework:** PyTorch
* **Epochs:** 200
* **Batch Size:** 4
* **Learning Rate:** 2e-4
* **Optimizer:** Adam
* **Scheduler:** Cosine Annealing

### 📉 Loss Function

Combined loss:

* MSE Loss
* SSIM Loss
* Perceptual Color Loss (LAB space)

---

## 📊 Dataset

* **UIEB Dataset (Underwater Image Enhancement Benchmark)**
* Total images: **890 paired samples** 
* Train: 801
* Validation: 89

---

## 📈 Evaluation Metrics

The model is evaluated using:

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)
* MSE (Mean Squared Error)
* UIQM (Underwater Image Quality Measure)
* UCIQE (Underwater Colour Image Quality Evaluation)

---

## 🚀 Results

### 🔥 Final Performance (Validation Set)

| Metric | Value        |
| ------ | ------------ |
| PSNR   | **24.10 dB** |
| SSIM   | **0.9192**   |
| MSE    | **0.0067**   |
| UIQM   | **44.39**    |
| UCIQE  | **28.61**    |

✔ PSNR improvement: **+5.89 dB over raw images** 

---

## 📊 Visual Outputs

The notebook includes:

* Training vs Validation Loss curves
* PSNR & SSIM graphs
* UIQM & UCIQE trends
* Stage-wise comparison (Raw → Denoised → Enhanced)
* Visual comparison with ground truth

---

## 🛠️ Technologies Used

* Python 🐍
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* scikit-image

---

## ▶️ How to Run

### Run on Kaggle

1. Open notebook
2. Click **Run All**

### Run Locally

```bash
pip install torch torchvision scikit-image matplotlib opencv-python
jupyter notebook
```

---

## 💡 Key Insights

* Combining **denoising + enhancement** improves performance significantly
* LAB-based color loss stabilizes color correction
* Residual learning helps preserve structural details
* Multi-metric evaluation provides better quality assessment

---

## 🔮 Future Improvements

* Use GAN-based architectures (e.g., UGAN, WaterNet)
* Real-time deployment
* Mobile/edge optimization
* Larger dataset training

---

## 📎 Author

**Paulette Gudapati**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

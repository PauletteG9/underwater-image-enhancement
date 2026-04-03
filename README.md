# Hybrid Underwater Image Enhancement
### CNN Blind Denoising + Perceptual Colour Enhancement Pipeline

---

## Overview

Underwater images suffer from wavelength-dependent colour attenuation, volumetric scattering, low contrast, and sensor noise. This project proposes a **two-stage hybrid deep learning pipeline** that first removes noise blindly, then corrects colour and contrast using a perceptual LAB colour loss.

The key insight: applying enhancement directly to a noisy image amplifies the noise. By decoupling denoising from enhancement, each network can specialise — leading to measurably better results.

---

## Results

| Method | PSNR (dB) | SSIM | MSE |
|---|---|---|---|
| Traditional (Hist. EQ.) | 16.80 | 0.620 | — |
| CNN Enhancement Only | 19.90 | 0.740 | — |
| **Proposed Hybrid (Ours)** | **24.42** | **0.724** | **0.0036** |

Additional metrics on UIEB validation set:
- **UIQM**: 1.635 (Underwater Image Quality Measure)
- **UCIQE**: 23.71 (Underwater Colour Image Quality Evaluation)

---

## Architecture

```
Raw Underwater Image
        │
        ▼
┌─────────────────────┐
│    BlindDenoiser     │  DnCNN-style · depth 8 · 64ch · ~317K params
│  out = clamp(x−η̂)  │  Learns noise residual, subtracts it
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    ColorEnhancer    │  6× ResBlock + SE Attention · 64ch · ~738K params
│  out = clamp(f+x)   │  Fixes colour/contrast, global skip connection
└─────────────────────┘
        │
        ▼
  Enhanced Output
```

**Total parameters: ~1.05M**

### BlindDenoiser
- DnCNN-style residual architecture
- Predicts noise map η̂, then: `output = clamp(x − η̂, 0, 1)`
- "Blind" — no prior knowledge of noise level required

### ColorEnhancer
- Head: Conv(3→64) + ReLU
- Body: 6× ResBlock — `F_out = ReLU(F_in + BN2(Conv2(ReLU(BN1(Conv1(F_in))))))`
- SE Attention: AvgPool → FC(64→16) → ReLU → FC(16→64) → Sigmoid
- Tail: Conv(64→3) + global input skip
- Output: `clamp(Conv_tail(F_se) + x, 0, 1)`

---

## Loss Function

```
L = L_pixel + 0.5 × L_ssim + 0.05 × L_color
```

| Term | Description | Weight |
|---|---|---|
| `L_pixel` | RGB MSE between output and reference | 1.0 |
| `L_ssim` | Differentiable SSIM loss (Gaussian kernel 11×11, σ=1.5) | 0.5 |
| `L_color` | MSE in normalised CIE LAB space — targets blue-green cast | 0.05 |

> LAB values are normalised (L/100, a/256, b/256) before computing loss to keep scale comparable to pixel MSE.

---

## Dataset

[UIEB — Underwater Image Enhancement Benchmark](https://li-chongyi.github.io/proj_benchmark.html)

- 890 matched raw ↔ reference image pairs
- Diverse scenes: reef, marine, murky, deep water
- Split: 90% train / 10% validation (seed 42)
- Preprocessing: resize to 256×256, normalise to [0, 1]

**Expected folder structure on Google Drive:**
```
MyDrive/
└── underwater-image-enhancement/
    ├── raw-890/
    │   └── raw-890/
    │       ├── img1.png
    │       └── ...
    └── reference-890/
        └── reference-890/
            ├── img1.png
            └── ...
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 4 |
| Epochs | 200 |
| Optimizer | Adam |
| Learning rate | 2×10⁻⁴ |
| LR scheduler | Cosine annealing (η_min = 1×10⁻⁵) |
| Weight decay | 1×10⁻⁵ |
| Gradient clip | 0.5 |
| Device | CUDA / CPU (auto) |

---

## Requirements

```
torch
torchvision
scikit-image
Pillow
numpy
matplotlib
```

Install with:
```bash
pip install torch torchvision scikit-image Pillow numpy matplotlib
```

Or on Google Colab (already in notebook):
```python
!pip install -q scikit-image torch torchvision
```

---

## Usage

### 1. Clone the repository
```bash
git clone https://github.com/PauletteG9/underwater-image-enhancement.git
cd underwater-image-enhancement
```

### 2. Open in Google Colab
Upload `underwater_image_enhancement_v3.ipynb` to Colab, or open directly from GitHub.

### 3. Mount Google Drive and set paths
```python
from google.colab import drive
drive.mount('/content/drive')

RAW_DIR = r'/content/drive/MyDrive/underwater-image-enhancement/raw-890/raw-890'
REF_DIR = r'/content/drive/MyDrive/underwater-image-enhancement/reference-890/reference-890'
```

### 4. Run all cells
The notebook will:
1. Train the BlindDenoiser + ColorEnhancer jointly
2. Save the best model checkpoint as `best_model.pth`
3. Generate 6 evaluation figures (loss curves, PSNR/SSIM, UIQM/UCIQE, stage-wise comparison, visual comparison, summary card)

### 5. Load saved model for inference
```python
checkpoint = torch.load('best_model.pth', map_location=DEVICE)
denoiser.load_state_dict(checkpoint['denoiser'])
enhancer.load_state_dict(checkpoint['enhancer'])

denoiser.eval()
enhancer.eval()

with torch.no_grad():
    denoised = denoiser(raw_image)
    enhanced = enhancer(denoised)
```

---

## Output Figures

| Figure | Description |
|---|---|
| `fig1_loss.png` | Training vs validation loss curves |
| `fig2_psnr_ssim.png` | PSNR and SSIM over epochs |
| `fig3_uiqm_uciqe.png` | UIQM and UCIQE over epochs |
| `fig4_stagewise.png` | Stage-wise metric comparison (Raw → Denoiser → Enhancer) |
| `fig5_visual.png` | Visual comparison of sample validation image |
| `fig6_summary.png` | Final performance summary card |

---

## Project Structure

```
underwater-image-enhancement/
│
├── underwater_image_enhancement_v3.ipynb   # Main notebook
├── best_model.pth                          # Saved model weights (after training)
├── fig1_loss.png                           # Generated figures
├── fig2_psnr_ssim.png
├── fig3_uiqm_uciqe.png
├── fig4_stagewise.png
├── fig5_visual.png
├── fig6_summary.png
└── README.md
```

---

## References

1. Peng et al., "U-Shape Transformer for Underwater Image Enhancement," IEEE TIP, 2023
2. Wang et al., "Simultaneous Restoration and Super-Resolution GAN," Frontiers in Marine Science, 2023
3. Tolie et al., "DICAM: Deep Inception and Channel-wise Attention Modules," Neurocomputing, 2024
4. Chandrasekar et al., "PhISH-Net," IEEE WACV, 2024
5. Li et al., "Dual High-Order Total Variation Model," IEEE TMM, 2024
6. Adagale-Vairagar et al., "Underwater Image Enhancement using Convolution Denoising Network," ETASR, 2025

---

## License

This project is submitted as academic work for M.Tech Data Science. For research and educational use only.

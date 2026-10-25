# ThermoSR: Research Methodology & Deep Learning Architecture

## 1. Abstract & Research Problem

Thermal infrared (TIR) cameras capture radiant thermal flux, enabling imaging across night-time, degraded visual environments (fog, smoke), and industrial monitoring. However, commercial long-wave infrared (LWIR) microbolometers suffer from low spatial resolution (typically $80 \times 60$ or $160 \times 120$) due to physical diffraction limits and high sensor manufacturing costs.

In contrast, optical visible cameras (RGB) offer high spatial resolution ($1280 \times 1024$ or $4K$) with rich high-frequency textures. Guided thermal super-resolution leverages this high-resolution RGB imagery to guide the reconstruction of fine edges and structures in low-resolution thermal observations.

### The Fundamental Guidance Dilemma
Standard RGB guidance assumes cross-modal spatial correlation. However:
1. **Sensor Parallax & Misalignment**: Multi-camera baselines produce parallax shifts between optical and thermal lenses.
2. **Modality Inconsistency**: Visual textures (e.g. shadow contours, surface paint, printed lettering) frequently exhibit no thermal gradient. Blindly injecting RGB features transfers hallucinated texture artifacts into the thermal output.
3. **Degraded Guidance**: In low-light or smoke scenarios, RGB imagery is noisy while thermal signals remain clear.

**ThermoSR** introduces **Reliability-Aware RGB Guidance**: a learned spatial reliability gate that estimates pixel-level cross-modal agreement, dynamically controlling RGB feature fusion to protect thermal physical fidelity.

---

## 2. Model Architectures

### Model A: Thermal-Only CNN Baseline
Establishes the fundamental performance baseline using only low-resolution thermal input:
$$I_T^{HR} = \mathcal{F}_{\text{CNN}}(I_T^{LR}) + \text{BicubicUpsample}(I_T^{LR}, s)$$

- **Trunk**: 8 Residual Convolutional Blocks with PReLU activations and residual channel scaling ($0.1$) to stabilize gradients.
- **Upsampling Head**: Cascaded PixelShuffle modules ($2\times \to 2\times \to 2\times$ for $\times 8$).
- **Output**: 1-channel high-resolution thermal intensity bounded strictly to $[0.0, 1.0]$.

### Model B: Standard RGB-Guided Thermal Fusion
Implements standard cross-modal feature concatenation:
- **Thermal Branch**: Residual encoder extracting features $F_T \in \mathbb{R}^{B \times 64 \times sH \times sW}$.
- **RGB Guidance Branch**: 3-channel optical encoder extracting features $F_{RGB} \in \mathbb{R}^{B \times 64 \times sH \times sW}$.
- **Fusion Trunk**: Concatenation $[F_T, F_{RGB}] \in \mathbb{R}^{B \times 128 \times sH \times sW} \to \text{Conv}_{1\times 1} \to \text{ResBlocks}$.
- **Head**: 1-channel reconstruction with global bicubic thermal skip connection.

### Model C: Proposed Reliability-Aware RGB Guidance
Introduces the **Spatial Reliability Gate** $G_{\text{rel}}$:
1. **Feature Alignment Projection**:
   $$P_T = \text{Conv}_{1\times 1}(F_T), \quad P_{RGB} = \text{Conv}_{1\times 1}(F_{RGB})$$
   $$D = |P_T - P_{RGB}|$$
2. **Confidence Estimation**:
   $$R(x, y) = \sigma(\text{Conv}_{3\times 3}(\text{PReLU}(\text{Conv}_{3\times 3}([P_T, P_{RGB}, D]))))$$
   where $R(x, y) \in [0.0, 1.0]$ represents the spatial confidence map.
3. **Adaptive Modality Fusion**:
   $$F_{RGB}^{\text{gated}} = \text{Conv}_{3\times 3}(F_{RGB} \odot R)$$
   $$F_{\text{fused}} = F_T + \text{ResBlocks}(\text{Conv}_{1\times 1}([F_T, F_{RGB}^{\text{gated}}]))$$
4. **Reconstructed Output & Heatmap**:
   $$\hat{I}_T = \text{Bicubic}(I_T^{LR}) + \text{Tail}(F_{\text{fused}})$$

---

## 3. Loss Functions & Physical Consistency

The objective function optimizes reconstruction fidelity while conserving physical thermal flux:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{L1} + \lambda_{\text{grad}} \mathcal{L}_{\text{grad}} + \lambda_{\text{consist}} \mathcal{L}_{\text{consist}}$$

1. **Reconstruction Loss ($L_1$)**:
   $$\mathcal{L}_{L1} = \frac{1}{HW} \sum_{i,j} |\hat{I}_T(i, j) - I_T^{HR}(i, j)|$$
2. **Thermal Gradient Edge Loss ($\mathcal{L}_{\text{grad}}$)**:
   Uses Sobel filters $S_x, S_y$ to enforce sharp thermal boundaries:
   $$\mathcal{L}_{\text{grad}} = \| S_x * \hat{I}_T - S_x * I_T^{HR} \|_1 + \| S_y * \hat{I}_T - S_y * I_T^{HR} \|_1$$
3. **Thermal Downsampling Cycle Consistency ($\mathcal{L}_{\text{consist}}$)**:
   Enforces physical conservation of energy across spatial scales:
   $$\mathcal{L}_{\text{consist}} = \| \text{Downsample}(\hat{I}_T, \text{scale}=1/s) - I_T^{LR} \|_1$$

---

## 4. Evaluation Metrics & Protocols

All quantitative evaluations are computed using standardized data ranges without synthetic inflation:
- **PSNR (dB)**: $20 \log_{10}(1.0 / \sqrt{\text{MSE}})$.
- **SSIM**: Calculated using structural similarity with dynamic range $1.0$.
- **LPIPS**: Perceptual patch distance computed using AlexNet feature activations.
- **Thermal Consistency L1**: Mean absolute discrepancy between downsampled prediction and original input LR measurement.

---

## 5. Assumptions & Limitations

1. **Calibrated Temperature vs Pixel Intensity**:
   Standard FLIR challenge and CIDIS JPEG images represent normalized thermal intensity $[0, 1]$ rather than radiometrically calibrated temperature in Kelvin/Celsius. Predictions must be presented as normalized thermal intensity unless camera calibration parameters (FLIR Planck constants $R, B, F, O$) are provided.
2. **Aspect Ratio Correspondence**:
   FLIR thermal cameras have native $4:3$ aspect ratios ($640 \times 480$), whereas optical cameras often have $5:4$ ($1280 \times 1024$) or $16:9$. Center cropping is applied symmetrically to align coordinate frames.

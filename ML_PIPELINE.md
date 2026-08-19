# DeepForensics — Machine Learning Pipeline Architecture

The DeepForensics machine learning pipeline implements a multidomain, multi-signal detection approach for identifying AI-generated and manipulated images and videos.

---

## 1. Domain Detection Models

### A. Spatial RGB Domain (`EfficientNetRGBModel`)
- **Architecture**: EfficientNet-B4 pre-trained on ImageNet and fine-tuned on synthetic media datasets.
- **Input**: 3-channel RGB image tensor normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Purpose**: Detects high-level semantic anomalies, anatomical distortions, lighting inconsistencies, and spatial blending artifacts.

### B. Frequency Domain (`ConvNeXtFrequencyModel`)
- **Architecture**: ConvNeXt-Tiny operating on 2D Discrete Cosine Transform (DCT) log-magnitude spectrums.
- **Input**: Log-magnitude 2D DCT spectrum map extracted from gray/RGB channels.
- **Purpose**: Detects transposed convolution grid patterns, spectral power drop-offs, and high-frequency checkerboard artifacts characteristic of diffusion upsamplers (e.g., Stable Diffusion, SDXL) and GAN generators.

### C. Noise Residual Domain (`SRMResidualModel`)
- **Architecture**: 3-filter SRM (Stegananalytic Rich Model) high-pass residual filter bank coupled with ResNet-18 feature extraction backbone.
- **Input**: High-pass residual noise maps.
- **Purpose**: Suppresses semantic content to isolate low-level pixel noise residuals, camera PRNU (Photo-Response Non-Uniformity) variations, and denoising signatures.

### D. Face Region Domain (`FaceDetector`)
- **Architecture**: OpenCV Haar Cascade face detector with bounding box alignment.
- **Purpose**: Detects faces, crops facial regions, and evaluates facial manipulation artifacts. Reports `face_status: "not_applicable"` if no face is detected in the image.

---

## 2. Dynamic Fusion & Calibration Engine

### A. Multidomain Score Fusion (`ScoreFusionEngine`)
Combines individual domain probability outputs:
$$P_{fused}(\text{AI}) = \sum_{d \in D_{active}} w_d \cdot P_d(\text{AI})$$
where domain weights $w_d$ are dynamically re-normalized based on which checkpoints are active.

### B. Probability Calibration (`ProbabilityCalibrator`)
Applies Platt scaling to raw model logits:
$$P_{calibrated}(\text{AI}) = \frac{1}{1 + \exp\left(-\left(\frac{\text{logit}}{T} + b\right)\right)}$$
where temperature $T$ and bias $b$ are fitted on validation splits to ensure probabilities correspond to empirical accuracy.

### C. Threshold Decision Engine (`ForensicDecisionEngine`)
Classifies images based on calibrated probability:
- $P(\text{AI}) \ge 0.65 \implies \mathbf{AI\_GENERATED}$
- $P(\text{AI}) \le 0.35 \implies \mathbf{REAL}$
- $0.35 < P(\text{AI}) < 0.65 \implies \mathbf{UNCERTAIN}$

---

## 3. Explainability Engine (`GradCAM`)

Gradient-weighted Class Activation Mapping generates 2D saliency maps overlaying the input image:
$$L_{GradCAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
Highlighting exact spatial regions (e.g. eyes, background, boundary edges) that contributed most to the AI classification.

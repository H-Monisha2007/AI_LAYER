export function Methodology() {
    const sections = [
        {
            emoji: '🔬',
            title: 'Detection Philosophy',
            body: `DeepForensics does not rely on any single model or heuristic. 
Instead, it uses a multidomain fusion strategy where independent models 
analyze complementary signal domains — RGB spatial, frequency spectrum, 
noise residual, face forgery, and temporal consistency — and their outputs 
are fused via a learned confidence-weighted ensemble.`,
        },
        {
            emoji: '🎨',
            title: 'RGB Spatial Domain (EfficientNet / ConvNeXt)',
            body: `EfficientNet-B4 and ConvNeXt spatial models learn texture and color statistics.
AI-generated images often have distinctive RGB patterns: pixel-level inconsistencies in eyes, 
hair, background blending, and over-smoothed skin tones that trained spatial CNNs can reliably 
detect. ConvNeXt uses depthwise separable convolutions for superior generalization.`,
        },
        {
            emoji: '📡',
            title: 'Frequency Domain (FFT / DCT)',
            body: `GAN generators produce frequency artifacts invisible to human eyes 
(periodic checkerboard patterns in FFT spectrum due to upsampling). Diffusion models 
leave different but consistent spectral signatures. Our frequency branch applies 
2D-DCT transformation and feeds the resulting spectrum into a ConvNeXt-Tiny model, 
capturing generative artifacts the spatial model cannot see.`,
        },
        {
            emoji: '🔊',
            title: 'Noise Residual Analysis (Vision Transformer)',
            body: `Camera sensor noise follows predictable statistical distributions (photon shot noise, 
read noise, demosaicing patterns). AI-generated images have near-zero sensor noise. 
The residual branch extracts high-frequency noise by subtracting a smoothed version 
of the image and feeds it to a ViT-Base/16 transformer which attends to residual 
anomaly patterns globally.`,
        },
        {
            emoji: '👤',
            title: 'Face Forgery Analysis',
            body: `Face-specific features (eyes, teeth, ears, hair edges) are common failure points 
for face swap and deepfake generators. When faces are detected (via a lightweight 
face detector), the face crop undergoes specialized model inference optimized for 
facial authentication artifacts. Results are incorporated into the domain fusion score.`,
        },
        {
            emoji: '⏱️',
            title: 'Temporal Analysis (CNN-LSTM / VideoMAE)',
            body: `Real videos have natural temporal continuity — blink patterns, micro-expressions, 
optical flow consistency. Deepfake videos typically show per-frame inconsistency in identity 
features. Our CNN-LSTM aggregates per-frame embeddings over time windows; 
VideoMAE/MViT models process 3D spatiotemporal patches for stronger detection.`,
        },
        {
            emoji: '⚖️',
            title: 'Multidomain Fusion & Calibration',
            body: `All domain scores are fed into a learnable fusion layer trained end-to-end. 
Calibration (temperature scaling or Platt scaling) is applied to ensure confidence 
outputs are statistically reliable. The system reports UNCERTAIN when ensemble model 
disagreement is high, rather than forcing a false confident prediction.`,
        },
        {
            emoji: '📊',
            title: 'Evaluation Protocol',
            body: `Evaluations are performed on FaceForensics++, Celeb-DF v2, and DFDC test sets 
with identity-aware (video-level) train/test splits to prevent frame-level data leakage. 
Metrics include Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, EER, FPR, and FNR 
at both frame-level and video-level. Cross-dataset generalization is reported separately.`,
        },
    ]

    return (
        <div className="main-content">
            <div className="page-header">
                <div className="page-title">📖 Methodology</div>
                <div className="page-subtitle">Technical approach, model architecture, and research design</div>
            </div>

            <div className="page-content">
                <div style={{ maxWidth: 800, display: 'flex', flexDirection: 'column', gap: 20 }}>
                    {sections.map(s => (
                        <div className="card animate-in" key={s.title}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                                <span style={{ fontSize: '1.8rem' }}>{s.emoji}</span>
                                <div style={{ fontWeight: 700, fontSize: '1rem' }}>{s.title}</div>
                            </div>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: 1.8 }}>
                                {s.body}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

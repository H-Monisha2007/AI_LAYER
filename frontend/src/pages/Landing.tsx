import { useNavigate } from 'react-router-dom'

export function Landing() {
    const nav = useNavigate()

    const features = [
        { icon: '🧠', title: 'Multidomain Fusion', desc: 'RGB, Frequency (DCT/FFT), Noise Residual and Face domain models fused for robust detection.' },
        { icon: '🎯', title: 'Grad-CAM Explainability', desc: 'Visual saliency maps highlight exactly which pixels drove the AI decision.' },
        { icon: '🎞️', title: 'Video Temporal Analysis', desc: 'CNN-LSTM and 3D-CNN models analyze frame sequences for deepfake artifacts over time.' },
        { icon: '📈', title: 'Research Dashboard', desc: 'Ablation experiments, ROC-AUC, EER, PR-AUC, and per-dataset cross-evaluation benchmarks.' },
        { icon: '⚡', title: 'GPU Accelerated', desc: 'Automatic CUDA detection. Falls back gracefully to CPU — never crashes the inference pipeline.' },
        { icon: '🔒', title: 'Secure & Validated', desc: 'MIME validation, extension checks, path traversal protection, and safe filename handling.' },
    ]

    return (
        <div className="hero">
            <div className="hero-eyebrow">Research-Grade AI Media Forensics</div>

            <h1 className="hero-title">
                Detect AI-Generated<br />Images & Videos
            </h1>

            <p className="hero-sub">
                DeepForensics is a multidomain deep learning framework combining spatial, frequency,
                residual, face, and temporal analysis models to expose synthetic media with explainable,
                calibrated confidence scores.
            </p>

            <div className="hero-ctas">
                <button className="btn btn-primary btn-lg" onClick={() => nav('/image')}>
                    🖼️ Analyze Image
                </button>
                <button className="btn btn-secondary btn-lg" onClick={() => nav('/video')}>
                    🎬 Analyze Video
                </button>
                <button className="btn btn-secondary btn-lg" onClick={() => nav('/evaluation')}>
                    📊 View Research Dashboard
                </button>
            </div>

            <div className="hero-features animate-in">
                {features.map(f => (
                    <div className="feature-card" key={f.title}>
                        <div className="feature-icon">{f.icon}</div>
                        <div className="feature-title">{f.title}</div>
                        <div className="feature-desc">{f.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    )
}

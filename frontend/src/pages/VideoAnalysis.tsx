import { useState, useCallback } from 'react'
import { detectVideo, DetectionSummary } from '../api'
import { ResultPanel } from '../components/ResultPanel'

export function VideoAnalysis() {
    const [file, setFile] = useState<File | null>(null)
    const [dragging, setDragging] = useState(false)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<DetectionSummary | null>(null)
    const [error, setError] = useState<string | null>(null)

    const handleFile = (f: File) => {
        setFile(f)
        setResult(null)
        setError(null)
    }

    const onDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setDragging(false)
        const f = e.dataTransfer.files[0]
        if (f) handleFile(f)
    }, [])

    const runDetection = async () => {
        if (!file) return
        setLoading(true)
        setError(null)
        try {
            const res = await detectVideo(file)
            setResult(res)
        } catch (e: any) {
            setError(e.message || 'Detection failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="main-content">
            <div className="page-header">
                <div className="page-title">🎬 Video Analysis</div>
                <div className="page-subtitle">Upload a video for temporal deepfake detection and frame-level analysis</div>
            </div>

            <div className="page-content">
                <div className="grid-2">
                    <div>
                        <label
                            className={`upload-zone${dragging ? ' drag-over' : ''}`}
                            onDragOver={e => { e.preventDefault(); setDragging(true) }}
                            onDragLeave={() => setDragging(false)}
                            onDrop={onDrop}
                            htmlFor="vid-upload"
                            style={{ display: 'block' }}
                        >
                            <div className="upload-icon">{file ? '✅' : '🎬'}</div>
                            <div className="upload-title">{file ? file.name : 'Drop video here'}</div>
                            <div className="upload-sub">{file ? `${(file.size / 1048576).toFixed(1)} MB` : 'or click to browse'}</div>
                            <div className="upload-hint">{file ? 'Ready to analyze' : 'MP4, AVI, MOV, MKV, WebM · max 100 MB'}</div>
                            <input id="vid-upload" type="file" accept=".mp4,.avi,.mov,.mkv,.webm" onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} style={{ display: 'none' }} />
                        </label>

                        {file && (
                            <div className="mt-4" style={{ display: 'flex', gap: 12 }}>
                                <button className="btn btn-primary" onClick={runDetection} disabled={loading} style={{ flex: 1 }}>
                                    {loading ? '⏳ Processing…' : '🎞️ Run Temporal Analysis'}
                                </button>
                                <button className="btn btn-secondary" onClick={() => { setFile(null); setResult(null) }}>✕ Clear</button>
                            </div>
                        )}

                        {file && (
                            <div className="card mt-4">
                                <div className="card-title">Pipeline Steps</div>
                                {['Validation & Metadata', 'Frame Sampling', 'Face Detection & Tracking', 'Per-frame RGB Model', 'Temporal CNN-LSTM', 'Domain Fusion', 'Suspicious Frame Detection', 'Confidence Timeline'].map((step, i) => (
                                    <div key={step} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '7px 0', borderBottom: 'none', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                                        <span style={{ color: 'var(--accent-blue)', fontWeight: 700, width: 20, fontSize: '0.75rem' }}>{i + 1}.</span>
                                        {step}
                                    </div>
                                ))}
                            </div>
                        )}

                        {error && (
                            <div className="mt-4 card" style={{ borderColor: 'var(--accent-red)', color: 'var(--accent-red)' }}>
                                ⚠️ {error}
                            </div>
                        )}
                    </div>

                    <div>
                        {loading && (
                            <div className="card" style={{ textAlign: 'center', padding: '60px 24px' }}>
                                <div className="spinner" />
                                <div style={{ marginTop: 16, color: 'var(--text-secondary)' }}>Running temporal analysis pipeline…</div>
                                <div style={{ marginTop: 8, color: 'var(--text-muted)', fontSize: '0.8rem' }}>Frame sampling → Face detection → Temporal fusion</div>
                            </div>
                        )}
                        {result && !loading && <ResultPanel result={result} />}
                        {!result && !loading && (
                            <div className="card" style={{ textAlign: 'center', padding: '60px 24px', color: 'var(--text-muted)' }}>
                                <div style={{ fontSize: '3rem', marginBottom: 12 }}>🎞️</div>
                                <div>Upload a video and click "Run Temporal Analysis"</div>
                                <div style={{ fontSize: '0.8rem', marginTop: 8 }}>Frame-level · Temporal · Confidence timeline</div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

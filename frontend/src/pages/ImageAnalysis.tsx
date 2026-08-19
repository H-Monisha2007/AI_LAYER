import { useState, useCallback } from 'react'
import { detectImage, DetectionSummary } from '../api'
import { ResultPanel } from '../components/ResultPanel'

export function ImageAnalysis() {
    const [file, setFile] = useState<File | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [dragging, setDragging] = useState(false)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<DetectionSummary | null>(null)
    const [error, setError] = useState<string | null>(null)

    const handleFile = (f: File) => {
        setFile(f)
        setResult(null)
        setError(null)
        const url = URL.createObjectURL(f)
        setPreview(url)
    }

    const onDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setDragging(false)
        const f = e.dataTransfer.files[0]
        if (f) handleFile(f)
    }, [])

    const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0]
        if (f) handleFile(f)
    }

    const runDetection = async () => {
        if (!file) return
        setLoading(true)
        setError(null)
        try {
            const res = await detectImage(file)
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
                <div className="page-title">🖼️ Image Analysis</div>
                <div className="page-subtitle">Upload an image for multidomain AI-generation detection</div>
            </div>

            <div className="page-content">
                <div className="grid-2">
                    <div>
                        <label
                            className={`upload-zone${dragging ? ' drag-over' : ''}`}
                            onDragOver={e => { e.preventDefault(); setDragging(true) }}
                            onDragLeave={() => setDragging(false)}
                            onDrop={onDrop}
                            htmlFor="img-upload"
                            style={{ display: 'block' }}
                        >
                            {preview ? (
                                <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 280, borderRadius: 12, objectFit: 'contain' }} />
                            ) : (
                                <>
                                    <div className="upload-icon">🖼️</div>
                                    <div className="upload-title">Drop image here</div>
                                    <div className="upload-sub">or click to browse</div>
                                    <div className="upload-hint">JPG, PNG, WebP, BMP · max 100 MB</div>
                                </>
                            )}
                            <input id="img-upload" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" onChange={onInputChange} style={{ display: 'none' }} />
                        </label>

                        {file && (
                            <div className="mt-4" style={{ display: 'flex', gap: 12 }}>
                                <button className="btn btn-primary" onClick={runDetection} disabled={loading} style={{ flex: 1 }}>
                                    {loading ? '⏳ Analyzing…' : '🔬 Run Detection'}
                                </button>
                                <button className="btn btn-secondary" onClick={() => { setFile(null); setPreview(null); setResult(null) }}>
                                    ✕ Clear
                                </button>
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
                                <div style={{ marginTop: 16, color: 'var(--text-secondary)' }}>Running multidomain analysis…</div>
                            </div>
                        )}
                        {result && !loading && <ResultPanel result={result} />}
                        {!result && !loading && (
                            <div className="card" style={{ textAlign: 'center', padding: '60px 24px', color: 'var(--text-muted)' }}>
                                <div style={{ fontSize: '3rem', marginBottom: 12 }}>🔬</div>
                                <div>Upload an image and click "Run Detection"</div>
                                <div style={{ fontSize: '0.8rem', marginTop: 8 }}>RGB · Frequency · Residual · Face fusion</div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

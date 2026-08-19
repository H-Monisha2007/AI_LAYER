import { useEffect, useState } from 'react'
import { fetchModels, ModelInfo } from '../api'

const domainColors: Record<string, string> = {
    spatial_rgb: 'var(--accent-blue)',
    frequency: 'var(--accent-cyan)',
    residual: 'var(--accent-purple)',
    face: 'var(--accent-orange)',
    temporal: 'var(--accent-green)',
    fusion: 'var(--accent-red)',
}

const domainEmoji: Record<string, string> = {
    spatial_rgb: '🎨',
    frequency: '📡',
    residual: '🔊',
    face: '👤',
    temporal: '⏱️',
    fusion: '⚡',
}

export function Models() {
    const [models, setModels] = useState<ModelInfo[]>([])
    const [loading, setLoading] = useState(true)
    const [err, setErr] = useState<string | null>(null)

    useEffect(() => {
        fetchModels()
            .then(setModels)
            .catch(e => setErr(e.message))
            .finally(() => setLoading(false))
    }, [])

    return (
        <div className="main-content">
            <div className="page-header">
                <div className="page-title">🤖 Model Registry</div>
                <div className="page-subtitle">Active deep learning models and their performance metrics</div>
            </div>

            <div className="page-content">
                {loading && <div style={{ textAlign: 'center', padding: 60 }}><div className="spinner" /></div>}
                {err && <div className="card" style={{ color: 'var(--accent-red)' }}>⚠️ {err}</div>}

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 20 }}>
                    {models.map(m => {
                        const color = domainColors[m.domain_type] || 'var(--accent-blue)'
                        const emoji = domainEmoji[m.domain_type] || '🤖'
                        return (
                            <div key={m.id} className="card" style={{ borderTop: `3px solid ${color}` }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                                    <span style={{ fontSize: '2rem' }}>{emoji}</span>
                                    <span className="tag" style={{ color, borderColor: color, background: `${color}18` }}>
                                        {m.domain_type.replace('_', ' ')}
                                    </span>
                                </div>

                                <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: 4 }}>{m.name}</div>
                                <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginBottom: 16 }}>
                                    {m.architecture} · v{m.version}
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                    {m.accuracy != null && (
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                                            <span style={{ color: 'var(--text-muted)' }}>Accuracy</span>
                                            <span style={{ color, fontWeight: 700 }}>{(m.accuracy * 100).toFixed(1)}%</span>
                                        </div>
                                    )}
                                    {m.roc_auc != null && (
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                                            <span style={{ color: 'var(--text-muted)' }}>ROC-AUC</span>
                                            <span style={{ color, fontWeight: 700 }}>{m.roc_auc.toFixed(3)}</span>
                                        </div>
                                    )}
                                    {m.eer != null && (
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                                            <span style={{ color: 'var(--text-muted)' }}>EER</span>
                                            <span style={{ fontWeight: 700 }}>{(m.eer * 100).toFixed(1)}%</span>
                                        </div>
                                    )}
                                </div>

                                <div style={{ marginTop: 16 }}>
                                    <span className={m.is_active ? 'badge badge-real' : 'badge'}>
                                        {m.is_active ? '✓ Active' : 'Inactive'}
                                    </span>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}

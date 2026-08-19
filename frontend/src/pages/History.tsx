import { useEffect, useState } from 'react'
import { fetchDetections, DetectionSummary } from '../api'

const BadgeMap: Record<string, string> = {
    REAL: 'badge-real',
    AI_GENERATED: 'badge-ai-generated',
    AI_MANIPULATED: 'badge-manipulated',
    UNCERTAIN: 'badge-uncertain',
}

export function History() {
    const [rows, setRows] = useState<DetectionSummary[]>([])
    const [loading, setLoading] = useState(true)
    const [err, setErr] = useState<string | null>(null)

    useEffect(() => {
        fetchDetections()
            .then(setRows)
            .catch(e => setErr(e.message))
            .finally(() => setLoading(false))
    }, [])

    return (
        <div className="main-content">
            <div className="page-header">
                <div className="page-title">📋 Detection History</div>
                <div className="page-subtitle">All past detections stored in the database</div>
            </div>

            <div className="page-content">
                <div className="card">
                    {loading && <div style={{ textAlign: 'center', padding: 40 }}><div className="spinner" /></div>}
                    {err && <div style={{ color: 'var(--accent-red)', padding: 24 }}>⚠️ {err}</div>}
                    {!loading && !err && rows.length === 0 && (
                        <div style={{ textAlign: 'center', padding: '60px 24px', color: 'var(--text-muted)' }}>
                            <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>📂</div>
                            No detections yet. Upload an image or video to get started.
                        </div>
                    )}
                    {!loading && rows.length > 0 && (
                        <div className="table-wrap">
                            <table>
                                <thead>
                                    <tr>
                                        <th>File</th>
                                        <th>Type</th>
                                        <th>Prediction</th>
                                        <th>Confidence</th>
                                        <th>RGB</th>
                                        <th>Freq.</th>
                                        <th>Residual</th>
                                        <th>Time (ms)</th>
                                        <th>Date</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rows.map(r => (
                                        <tr key={r.id}>
                                            <td title={r.original_filename} style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                {r.original_filename}
                                            </td>
                                            <td><span className="tag">{r.media_type}</span></td>
                                            <td>
                                                <span className={`badge ${BadgeMap[r.primary_prediction] || ''}`}>
                                                    {r.primary_prediction.replace('_', ' ')}
                                                </span>
                                            </td>
                                            <td style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>
                                                {r.confidence != null ? `${Math.round(r.confidence * 100)}%` : 'N/A'}
                                            </td>
                                            <td>{r.rgb_score != null ? `${Math.round(r.rgb_score * 100)}%` : '—'}</td>
                                            <td>{r.frequency_score != null ? `${Math.round(r.frequency_score * 100)}%` : '—'}</td>
                                            <td>{r.residual_score != null ? `${Math.round(r.residual_score * 100)}%` : '—'}</td>
                                            <td>{r.processing_time_ms != null ? r.processing_time_ms.toFixed(0) : '—'}</td>
                                            <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                                                {new Date(r.created_at).toLocaleString()}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

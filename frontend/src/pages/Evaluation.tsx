import { useEffect, useState } from 'react'
import { fetchEvaluationData, EvaluationReport } from '../api'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Legend
} from 'recharts'

export function Evaluation() {
    const [report, setReport] = useState<EvaluationReport | null>(null)
    const [loading, setLoading] = useState(true)
    const [err, setErr] = useState<string | null>(null)

    useEffect(() => {
        fetchEvaluationData().then(setReport).catch(e => setErr(e.message)).finally(() => setLoading(false))
    }, [])

    if (loading) return (
        <div className="main-content">
            <div className="page-header"><div className="page-title">📊 Forensic Evaluation & Benchmarks</div></div>
            <div className="page-content" style={{ textAlign: 'center', paddingTop: 80 }}><div className="spinner" /></div>
        </div>
    )

    if (err || !report) return (
        <div className="main-content">
            <div className="page-header"><div className="page-title">📊 Forensic Evaluation & Benchmarks</div></div>
            <div className="page-content"><div className="card" style={{ color: 'var(--accent-red)' }}>⚠️ {err || 'No evaluation data available'}</div></div>
        </div>
    )

    const overall = report.overall_metrics || {}
    const unseen = report.unseen_generator_test || {}
    const perGen = report.per_generator_performance || []
    const robustness = report.robustness_benchmarks || {}
    const failureAnalysis = report.failure_analysis || {}

    const robustnessData = Object.entries(robustness)
        .filter(([_, val]) => val !== null)
        .map(([key, val]) => ({
            degradation: key.replace('_', ' ').toUpperCase(),
            accuracy: +((val as number) * 100).toFixed(1)
        }))

    return (
        <div className="main-content">
            <div className="page-header">
                <div className="page-title">📊 Scientific Evaluation & Benchmark Suite</div>
                <div className="page-subtitle">Genuine evaluation metrics computed across generator-holdout tests and image degradation stress suites</div>
            </div>

            <div className="page-content">
                {report.status === 'MODEL_NOT_READY' && (
                    <div className="card mb-6" style={{ borderColor: 'var(--accent-orange)', color: 'var(--accent-orange)' }}>
                        ⚠️ Model checkpoints not trained. Run training to generate evaluation benchmarks.
                    </div>
                )}

                {/* Stat cards */}
                <div className="stats-grid mb-6">
                    <div className="stat-card">
                        <div className="stat-label">Overall Accuracy</div>
                        <div className="stat-value blue">{overall.accuracy != null ? `${(overall.accuracy * 100).toFixed(1)}%` : '—'}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">F1-Score</div>
                        <div className="stat-value green">{overall.f1_score != null ? `${(overall.f1_score * 100).toFixed(1)}%` : '—'}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">ROC-AUC</div>
                        <div className="stat-value purple">{overall.roc_auc != null ? overall.roc_auc.toFixed(3) : '—'}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">Equal Error Rate (EER)</div>
                        <div className="stat-value red">{overall.eer != null ? `${(overall.eer * 100).toFixed(1)}%` : '—'}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">Precision / Recall</div>
                        <div className="stat-value cyan" style={{ fontSize: '1.3rem' }}>
                            {overall.precision != null ? `${(overall.precision * 100).toFixed(0)}%` : '—'} / {overall.recall != null ? `${(overall.recall * 100).toFixed(0)}%` : '—'}
                        </div>
                    </div>
                </div>

                <div className="grid-2 mb-6">
                    {/* Unseen Generator Holdout Card */}
                    <div className="card">
                        <div className="card-title">🛡️ Unseen Generator Holdout Test</div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 16 }}>
                            Evaluates generalizability against unseen architectures (Midjourney, FLUX) excluded from training.
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, textAlign: 'center' }}>
                            <div style={{ background: 'var(--bg-secondary)', padding: 12, borderRadius: 8 }}>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Accuracy</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-blue)', marginTop: 4 }}>
                                    {unseen.accuracy != null ? `${(unseen.accuracy * 100).toFixed(1)}%` : '—'}
                                </div>
                            </div>
                            <div style={{ background: 'var(--bg-secondary)', padding: 12, borderRadius: 8 }}>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>F1 Score</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-green)', marginTop: 4 }}>
                                    {unseen.f1_score != null ? `${(unseen.f1_score * 100).toFixed(1)}%` : '—'}
                                </div>
                            </div>
                            <div style={{ background: 'var(--bg-secondary)', padding: 12, borderRadius: 8 }}>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>ROC-AUC</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-purple)', marginTop: 4 }}>
                                    {unseen.roc_auc != null ? unseen.roc_auc.toFixed(3) : '—'}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Robustness Stress Benchmark Bar Chart */}
                    <div className="card">
                        <div className="card-title">⚡ Robustness Under Image Degradation</div>
                        {robustnessData.length === 0 ? (
                            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>No robustness benchmarks recorded</div>
                        ) : (
                            <ResponsiveContainer width="100%" height={220}>
                                <BarChart data={robustnessData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                    <XAxis dataKey="degradation" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
                                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                                    <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-primary)' }} />
                                    <Bar dataKey="accuracy" fill="var(--accent-cyan)" radius={[4, 4, 0, 0]} name="Accuracy %" />
                                </BarChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                {/* Per Generator Table */}
                {perGen.length > 0 && (
                    <div className="card mb-6">
                        <div className="card-title">🎯 Per-Generator Performance Breakdown</div>
                        <div className="table-wrap">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Generator Source</th>
                                        <th>Sample Count</th>
                                        <th>Accuracy</th>
                                        <th>F1 Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {perGen.map((g, idx) => (
                                        <tr key={idx}>
                                            <td style={{ fontWeight: 600 }}>{g.generator}</td>
                                            <td>{g.count}</td>
                                            <td style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>{(g.accuracy * 100).toFixed(1)}%</td>
                                            <td style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{(g.f1_score * 100).toFixed(1)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Failure Analysis Table */}
                {failureAnalysis.false_negatives_samples && failureAnalysis.false_negatives_samples.length > 0 && (
                    <div className="card">
                        <div className="card-title">🔍 Scientific Failure Analysis (False Negatives & False Positives)</div>
                        <div className="table-wrap">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Sample Filename</th>
                                        <th>Error Type</th>
                                        <th>Predicted P(AI)</th>
                                        <th>Likely Failure Root Cause</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {failureAnalysis.false_negatives_samples.map((s: any, idx: number) => (
                                        <tr key={idx}>
                                            <td style={{ fontSize: '0.82rem', fontFamily: 'monospace' }}>{s.filename}</td>
                                            <td><span className="tag" style={{ background: 'rgba(239, 68, 68, 0.15)', color: 'var(--accent-red)' }}>{s.error_type}</span></td>
                                            <td style={{ fontWeight: 700 }}>{(s.predicted_p_ai * 100).toFixed(1)}%</td>
                                            <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{s.likely_cause}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

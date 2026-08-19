import { useState } from 'react'
import { DetectionSummary, VerdictWarning } from '../api'

const GLOSSARY: Record<string, string> = {
    'AI-Generation Score': 'Estimates how strongly the forensic models favor AI-generated imagery based on the available evidence.',
    'Real-Image Score': 'Estimates how strongly the forensic models favor naturally captured or non-AI imagery based on the available evidence.',
    'Confidence': 'Indicates how strongly the model favors its current prediction relative to the decision boundary.',
    'Reliability': 'Indicates how trustworthy the prediction is given model agreement, image quality, calibration, and domain compatibility.',
    'Model Agreement': 'Measures how consistently the independent forensic models respond to the same image.',
    'OOD Status': 'Indicates whether the image falls within the types of visual data on which the detection system was calibrated.',
    'Uncertainty Margin': 'Represents the range around the prediction where the available evidence does not clearly favor one class.',

    'RGB Spatial Analysis': 'Examines visual structures, textures, edges, and spatial patterns that may provide supporting evidence of synthetic image generation.',
    'Frequency DCT Analysis': 'Examines frequency-domain characteristics that may provide supporting evidence of synthetic image generation.',
    'Noise Residual Analysis': 'Examines subtle pixel-level noise patterns that may provide supporting evidence about how the image was produced.',
    'Patch Analysis': 'Examines smaller regions of the image to identify localized patterns that may contribute to the overall forensic assessment.',

    'Spectral Energy Ratio': 'Measures how image information is distributed across frequency components and provides supporting forensic evidence.',
    'Noise Variance': 'Measures the variation within the extracted image noise and provides supporting evidence about low-level image characteristics.',
    'Patch Max P(AI)': 'Represents the strongest AI-generation score observed among the analyzed image regions.',
    'Patch Mean P(AI)': 'Represents the average AI-generation score across the analyzed image regions.',
    'Patch Variance': 'Measures how much AI-generation scores differ across the analyzed image regions.',
    'EXIF Metadata': 'Contains embedded information about an image\'s capture or processing history when such metadata is available.',
    'JPEG Quantization': 'Examines JPEG compression parameters that provide supporting information about the image\'s encoding history.',

    'Metadata Absent': 'No embedded metadata was found; this is common after image sharing or processing and is not evidence of AI generation by itself.',
    'Model Disagreement Warning': 'Independent forensic models produced substantially different predictions, reducing the reliability of the overall assessment.',
    'OOD Warning': 'The image differs from the data characteristics represented during model calibration, so the prediction may be less reliable.',
}

const verdictStyles: Record<string, { bg: string; border: string; color: string; icon: string }> = {
    HIGH_CHANCE_AI_GENERATED: {
        bg: 'rgba(239, 68, 68, 0.12)',
        border: 'rgba(239, 68, 68, 0.5)',
        color: '#ef4444',
        icon: '🔴',
    },
    MODERATE_CHANCE_AI_GENERATED: {
        bg: 'rgba(245, 158, 11, 0.12)',
        border: 'rgba(245, 158, 11, 0.5)',
        color: '#f59e0b',
        icon: '🟠',
    },
    UNCERTAIN: {
        bg: 'rgba(234, 179, 8, 0.12)',
        border: 'rgba(234, 179, 8, 0.5)',
        color: '#eab308',
        icon: '🟡',
    },
    MODERATE_CHANCE_REAL: {
        bg: 'rgba(34, 197, 94, 0.12)',
        border: 'rgba(34, 197, 94, 0.5)',
        color: '#22c55e',
        icon: '🟢',
    },
    HIGH_CHANCE_REAL: {
        bg: 'rgba(34, 197, 94, 0.12)',
        border: 'rgba(34, 197, 94, 0.5)',
        color: '#22c55e',
        icon: '🟢',
    },
}

function reliabilityLabel(r: number): { text: string; color: string } {
    if (r >= 0.75) return { text: 'High', color: '#22c55e' }
    if (r >= 0.50) return { text: 'Moderate', color: '#f59e0b' }
    if (r >= 0.30) return { text: 'Low', color: '#ef4444' }
    return { text: 'Very Low', color: '#dc2626' }
}

function oodLabel(status: string): { text: string; color: string } {
    if (status === 'OK') return { text: 'OK', color: '#22c55e' }
    if (status === 'CAUTION') return { text: 'Caution', color: '#f59e0b' }
    return { text: 'Warning', color: '#ef4444' }
}

interface Props { result: DetectionSummary }

function InfoTooltip({ text }: { text: string }) {
    const [visible, setVisible] = useState(false)
    return (
        <span
            style={{ position: 'relative', display: 'inline-block', marginLeft: 4, cursor: 'help' }}
            onMouseEnter={() => setVisible(true)}
            onMouseLeave={() => setVisible(false)}
        >
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', border: '1px solid var(--border-subtle)', borderRadius: '50%', padding: '0 4px' }}>?</span>
            {visible && (
                <div style={{
                    position: 'absolute',
                    bottom: '125%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: 220,
                    padding: '8px 12px',
                    background: '#1e293b',
                    color: '#f8fafc',
                    fontSize: '0.75rem',
                    borderRadius: 6,
                    boxShadow: '0 10px 15px -3px rgba(0,0,0,0.5)',
                    zIndex: 100,
                    lineHeight: 1.4,
                    pointerEvents: 'none',
                }}>
                    {text}
                </div>
            )}
        </span>
    )
}

function DomainBar({ label, tooltip, score, accentColor }: { label: string; tooltip?: string; score?: number | null; accentColor: string }) {
    if (score === undefined || score === null) return null
    const pct = Math.round(score * 100)
    return (
        <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                <span style={{ color: 'var(--text-secondary)' }}>
                    {label}
                    {tooltip && <InfoTooltip text={tooltip} />}
                </span>
                <span style={{ color: accentColor, fontWeight: 700 }}>{pct}% AI</span>
            </div>
            <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-tertiary)' }}>
                <div style={{ width: `${pct}%`, height: '100%', borderRadius: 3, background: accentColor, transition: 'width 0.6s ease' }} />
            </div>
        </div>
    )
}

function MetricPill({ label, value, color, tooltip }: { label: string; value: string; color: string; tooltip?: string }) {
    return (
        <div style={{ background: 'var(--bg-secondary)', borderRadius: 8, padding: '10px 12px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {label}
                {tooltip && <InfoTooltip text={tooltip} />}
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 800, color, marginTop: 2 }}>{value}</div>
        </div>
    )
}

export function ResultPanel({ result }: Props) {
    const verdict = result.verdict
    const verdictKey = verdict?.verdict_key || 'UNCERTAIN'
    const style = verdictStyles[verdictKey] || verdictStyles.UNCERTAIN
    const displayLabel = verdict?.display_label || 'UNCERTAIN / INSUFFICIENT EVIDENCE'
    const icon = verdict?.category_icon || style.icon

    const aiScore = verdict?.ai_score ?? result.calibrated_probability_ai ?? result.confidence ?? 0
    const confidence = verdict?.confidence ?? result.confidence ?? 0
    const reliability = verdict?.reliability ?? result.reliability ?? 0
    const modelAgreement = verdict?.model_agreement ?? result.model_agreement ?? 1.0
    const oodStatus = verdict?.ood_status ?? result.ood_status ?? 'OK'
    const explanation = verdict?.user_explanation ?? result.user_explanation ?? ''
    const warnings: VerdictWarning[] = result.warnings || []
    const evidence = result.evidence || {}

    const relInfo = reliabilityLabel(reliability)
    const oodInfo = oodLabel(oodStatus)

    return (
        <div className="result-panel">
            {/* Model Not Ready Banner */}
            {result.status === 'MODEL_NOT_READY' && (
                <div style={{ marginBottom: 20, padding: '14px 16px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid var(--accent-orange)', borderRadius: 8, fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                    <div style={{ fontWeight: 700, color: 'var(--accent-orange)', marginBottom: 4 }}>⚠️ Model Checkpoint Not Loaded</div>
                    <div>{result.message || 'Trained model weights are not installed. Please add PyTorch weights to model_weights/ directory.'}</div>
                </div>
            )}

            {/* ═══════ PRIMARY VERDICT CARD ═══════ */}
            <div style={{
                background: style.bg,
                border: `1.5px solid ${style.border}`,
                borderRadius: 12,
                padding: '28px 24px',
                marginBottom: 24,
                textAlign: 'center',
            }}>
                <div style={{ fontSize: '2.6rem', lineHeight: 1 }}>{icon}</div>
                <div style={{
                    fontSize: '1.35rem',
                    fontWeight: 900,
                    color: style.color,
                    marginTop: 10,
                    letterSpacing: '0.5px',
                    textTransform: 'uppercase',
                }}>
                    {displayLabel}
                </div>
                <div style={{ marginTop: 16, display: 'flex', justifyContent: 'center', gap: 32 }}>
                    <div>
                        <div style={{ fontSize: '2.4rem', fontWeight: 900, color: style.color, lineHeight: 1 }}>
                            {Math.round(aiScore * 100)}%
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                            AI-Generation Score
                            <InfoTooltip text={GLOSSARY['AI-Generation Score']} />
                        </div>
                    </div>
                    <div style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: 32 }}>
                        <div style={{ fontSize: '2.4rem', fontWeight: 900, color: relInfo.color, lineHeight: 1 }}>
                            {relInfo.text}
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                            Reliability
                            <InfoTooltip text={GLOSSARY['Reliability']} />
                        </div>
                    </div>
                </div>
            </div>

            {/* ═══════ EXPLANATION ═══════ */}
            {explanation && (
                <div style={{
                    marginBottom: 24,
                    padding: '14px 18px',
                    background: 'var(--bg-secondary)',
                    borderRadius: 10,
                    borderLeft: `3px solid ${style.color}`,
                }}>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 6 }}>Why This Result?</div>
                    <div style={{ fontSize: '0.88rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>{explanation}</div>
                </div>
            )}

            {/* ═══════ RELIABILITY METRICS ═══════ */}
            <div style={{ marginBottom: 24 }}>
                <div className="card-title" style={{ fontSize: '0.85rem', marginBottom: 12 }}>Reliability Assessment</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                    <MetricPill label="Confidence" value={`${Math.round(confidence * 100)}%`} color={style.color} tooltip={GLOSSARY['Confidence']} />
                    <MetricPill label="Reliability" value={`${Math.round(reliability * 100)}%`} color={relInfo.color} tooltip={GLOSSARY['Reliability']} />
                    <MetricPill label="Agreement" value={`${Math.round(modelAgreement * 100)}%`} color={modelAgreement >= 0.6 ? '#22c55e' : '#f59e0b'} tooltip={GLOSSARY['Model Agreement']} />
                    <MetricPill label="OOD Status" value={oodInfo.text} color={oodInfo.color} tooltip={GLOSSARY['OOD Status']} />
                </div>
            </div>

            {/* ═══════ MULTIDOMAIN FORENSIC SIGNALS ═══════ */}
            <div style={{ marginBottom: 24 }}>
                <div className="card-title" style={{ fontSize: '0.85rem', marginBottom: 12 }}>Multidomain Forensic Signals</div>
                <div style={{ background: 'var(--bg-secondary)', borderRadius: 10, padding: '16px 18px' }}>
                    <DomainBar label="RGB Spatial (EfficientNet-B4)" tooltip={GLOSSARY['RGB Spatial Analysis']} score={result.rgb_score} accentColor="var(--accent-blue)" />
                    <DomainBar label="Frequency DCT (ConvNeXt-Tiny)" tooltip={GLOSSARY['Frequency DCT Analysis']} score={result.frequency_score} accentColor="var(--accent-cyan)" />
                    <DomainBar label="Noise Residual (SRM-ResNet18)" tooltip={GLOSSARY['Noise Residual Analysis']} score={result.residual_score} accentColor="var(--accent-purple)" />
                    <DomainBar label="Face Crop Analysis" score={result.face_score} accentColor="var(--accent-orange)" />
                    <DomainBar label="Temporal Sequence" score={result.temporal_score} accentColor="var(--accent-green)" />
                </div>
            </div>

            {/* ═══════ DETAILED FORENSIC EVIDENCE ═══════ */}
            {Object.keys(evidence).length > 0 && (
                <div style={{ marginBottom: 24, padding: '16px 18px', background: 'var(--bg-secondary)', borderRadius: 10 }}>
                    <div className="card-title" style={{ fontSize: '0.85rem', marginBottom: 12 }}>Detailed Forensic Evidence</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: '0.82rem' }}>
                        {evidence.frequency && (
                            <div>
                                <span style={{ color: 'var(--text-muted)' }}>Spectral Energy Ratio: </span>
                                <strong>{((evidence.frequency.high_freq_energy_ratio ?? 0) * 100).toFixed(1)}%</strong>
                                <InfoTooltip text={GLOSSARY['Spectral Energy Ratio']} />
                            </div>
                        )}
                        {evidence.residual && (
                            <div>
                                <span style={{ color: 'var(--text-muted)' }}>Noise Variance: </span>
                                <strong>{typeof evidence.residual.noise_variance === 'number' ? evidence.residual.noise_variance.toFixed(4) : evidence.residual.noise_variance}</strong>
                                <InfoTooltip text={GLOSSARY['Noise Variance']} />
                            </div>
                        )}
                        {evidence.patch && (
                            <>
                                <div>
                                    <span style={{ color: 'var(--text-muted)' }}>Patch Max P(AI): </span>
                                    <strong>{Math.round((evidence.patch.patch_max_p_ai ?? 0) * 100)}%</strong>
                                    <InfoTooltip text={GLOSSARY['Patch Max P(AI)']} />
                                </div>
                                <div>
                                    <span style={{ color: 'var(--text-muted)' }}>Patch Variance: </span>
                                    <strong>{(evidence.patch.patch_variance ?? 0).toFixed(4)}</strong>
                                    <InfoTooltip text={GLOSSARY['Patch Variance']} />
                                </div>
                            </>
                        )}
                        {evidence.metadata && (
                            <>
                                <div>
                                    <span style={{ color: 'var(--text-muted)' }}>EXIF Header: </span>
                                    <strong>{evidence.metadata.exif_present ? 'Present' : 'Absent'}</strong>
                                    <InfoTooltip text={GLOSSARY['EXIF Metadata']} />
                                </div>
                                <div>
                                    <span style={{ color: 'var(--text-muted)' }}>JPEG Quantization: </span>
                                    <strong>{evidence.metadata.jpeg_quant_valid ? 'Valid Header' : 'N/A'}</strong>
                                    <InfoTooltip text={GLOSSARY['JPEG Quantization']} />
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}

            {/* ═══════ WARNINGS ═══════ */}
            {warnings.length > 0 && (
                <div style={{ marginBottom: 24 }}>
                    <div className="card-title" style={{ fontSize: '0.85rem', marginBottom: 10 }}>Warnings & Interpretation</div>
                    {warnings.map((w, idx) => (
                        <div key={idx} style={{
                            marginBottom: 8,
                            padding: '12px 16px',
                            background: 'rgba(245, 158, 11, 0.08)',
                            border: '1px solid rgba(245, 158, 11, 0.25)',
                            borderRadius: 8,
                        }}>
                            <div style={{ fontWeight: 700, fontSize: '0.84rem', color: '#f59e0b', marginBottom: 2 }}>⚠ {w.title}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>{w.message}</div>
                        </div>
                    ))}
                </div>
            )}

            {/* ═══════ LIMITATIONS ═══════ */}
            <div style={{
                marginBottom: 24,
                padding: '12px 16px',
                background: 'rgba(100, 116, 139, 0.08)',
                border: '1px solid rgba(100, 116, 139, 0.2)',
                borderRadius: 8,
                fontSize: '0.78rem',
                color: 'var(--text-muted)',
                lineHeight: 1.5,
            }}>
                <strong style={{ color: 'var(--text-secondary)' }}>ℹ Forensic Disclaimer:</strong> AI-image detection is probabilistic.
                Results represent forensic model estimates based on available evidence and should be interpreted as supporting evidence rather than absolute proof of image origin.
            </div>

            {/* ═══════ META GRID ═══════ */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {[
                    { label: 'File', value: result.original_filename },
                    { label: 'Type', value: result.media_type },
                    { label: 'Processing Time', value: result.processing_time_ms ? `${result.processing_time_ms.toFixed(0)} ms` : '—' },
                    { label: 'Detection ID', value: result.id.slice(0, 8) + '…' },
                ].map(({ label, value }) => (
                    <div key={label} style={{ background: 'var(--bg-secondary)', borderRadius: 8, padding: '10px 14px' }}>
                        <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px' }}>{label}</div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', marginTop: 2, wordBreak: 'break-all' }}>{value}</div>
                    </div>
                ))}
            </div>
        </div>
    )
}

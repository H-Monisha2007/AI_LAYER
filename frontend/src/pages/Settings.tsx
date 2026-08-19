export function Settings() {
    return (
        <div className="main-content">
            <div className="page-header">
                <div className="page-title">⚙️ Settings</div>
                <div className="page-subtitle">System configuration and model preferences</div>
            </div>
            <div className="page-content">
                <div className="grid-2">
                    <div className="card">
                        <div className="card-title">Backend Configuration</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 8 }}>
                            {[
                                { label: 'API Endpoint', value: 'http://localhost:8000/api', note: 'Set in .env' },
                                { label: 'Database', value: 'SQLite (dev) / PostgreSQL (prod)', note: 'Configured via DATABASE_URL' },
                                { label: 'Max Upload Size', value: '100 MB', note: 'MAX_UPLOAD_SIZE_MB' },
                                { label: 'Device', value: 'CPU (CUDA unavailable)', note: 'Auto-detected at startup' },
                            ].map(s => (
                                <div key={s.label} style={{ borderBottom: '1px solid var(--border)', paddingBottom: 12 }}>
                                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 2 }}>{s.label}</div>
                                    <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{s.value}</div>
                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>{s.note}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-title">Allowed Media Types</div>
                        <div style={{ marginTop: 12 }}>
                            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: '0.85rem' }}>Images</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
                                {['JPG', 'JPEG', 'PNG', 'WebP', 'BMP'].map(ext => (
                                    <span key={ext} className="tag">.{ext.toLowerCase()}</span>
                                ))}
                            </div>
                            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: '0.85rem' }}>Videos</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                                {['MP4', 'AVI', 'MOV', 'MKV', 'WebM'].map(ext => (
                                    <span key={ext} className="tag">.{ext.toLowerCase()}</span>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-title">Security Features</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
                            {[
                                'MIME type validation on every upload',
                                'File extension allowlist enforcement',
                                'Path traversal attack prevention',
                                'Filename sanitization (removes directory components)',
                                'File size limit (100 MB default)',
                                'SHA-256 content integrity hashing',
                                'Temporary file auto-cleanup',
                                'Media files never executed server-side',
                            ].map(f => (
                                <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                                    <span style={{ color: 'var(--accent-green)', fontSize: '0.7rem' }}>✓</span>
                                    {f}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-title">MLflow Tracking</div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                            MLflow experiment tracking is configured at <code style={{ background: 'var(--bg-secondary)', padding: '2px 6px', borderRadius: 4, fontSize: '0.78rem' }}>./mlruns</code>.
                            <br /><br />
                            To launch the MLflow UI:
                            <pre style={{ background: 'var(--bg-secondary)', padding: 12, borderRadius: 8, marginTop: 10, fontSize: '0.78rem', overflow: 'auto' }}>
                                python -m mlflow ui --port 5000
                            </pre>
                            Then open <strong>http://localhost:5000</strong> to view all experiment runs, metrics, and artifacts.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

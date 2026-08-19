import React, { Component, ErrorInfo, ReactNode } from 'react'

interface Props {
    children: ReactNode
}

interface State {
    hasError: boolean
    error: Error | null
    errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
        errorInfo: null,
    }

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error, errorInfo: null }
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Uncaught error caught by ErrorBoundary:', error, errorInfo)
        this.setState({ error, errorInfo })
    }

    public handleRetry = () => {
        this.setState({ hasError: false, error: null, errorInfo: null })
        window.location.reload()
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '100vh',
                    backgroundColor: '#0b0f19',
                    color: '#f3f4f6',
                    fontFamily: 'Inter, system-ui, sans-serif',
                    padding: '2rem',
                    textAlign: 'center',
                }}>
                    <div style={{
                        background: '#111827',
                        border: '1px solid #1f2937',
                        borderRadius: '12px',
                        padding: '2.5rem',
                        maxWidth: '600px',
                        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
                    }}>
                        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔬</div>
                        <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.75rem', color: '#f87171' }}>
                            DeepForensics Application Error
                        </h1>
                        <p style={{ color: '#9ca3af', fontSize: '0.9rem', marginBottom: '1.5rem', lineHeight: 1.5 }}>
                            DeepForensics encountered an unexpected UI rendering error. You can try refreshing the page or reloading the application.
                        </p>

                        <button
                            onClick={this.handleRetry}
                            style={{
                                backgroundColor: '#3b82f6',
                                color: '#ffffff',
                                border: 'none',
                                borderRadius: '8px',
                                padding: '0.75rem 1.5rem',
                                fontSize: '0.9rem',
                                fontWeight: 600,
                                cursor: 'pointer',
                                marginBottom: '1.5rem',
                                transition: 'background-color 0.2s',
                            }}
                        >
                            🔄 Reload Application
                        </button>

                        {this.state.error && (
                            <div style={{
                                textAlign: 'left',
                                backgroundColor: '#030712',
                                border: '1px solid #374151',
                                borderRadius: '8px',
                                padding: '1rem',
                                fontSize: '0.8rem',
                                fontFamily: 'monospace',
                                color: '#f87171',
                                maxHeight: '200px',
                                overflowY: 'auto',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-all',
                            }}>
                                <strong>Error:</strong> {this.state.error.toString()}
                            </div>
                        )}
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}

import { useState, useEffect } from 'react'

const API_BASE_URL = '/api'

export interface HealthData {
    status: string
    app_name: string
    version: string
    environment: string
    database_connected: boolean
    cuda_available: boolean
    device: string
    timestamp: string
}

export interface ModelInfo {
    id: string
    name: string
    version: string
    domain_type: string
    architecture: string
    accuracy?: number | null
    roc_auc?: number | null
    eer?: number | null
    is_active: boolean
}

export interface VerdictWarning {
    code: string
    title: string
    message: string
}

export interface VerdictDetail {
    verdict_key: string
    display_label: string
    category_icon: string
    ai_score: number
    real_score: number
    confidence: number
    reliability: number
    model_agreement: number
    ood_status: string
    user_explanation: string
}

export interface DetectionSummary {
    id: string
    media_file_id: string
    original_filename: string
    media_type: string
    status?: string
    primary_prediction: string
    decision?: string | null
    verdict?: VerdictDetail | null
    confidence?: number | null
    reliability?: number | null
    calibrated_probability_ai?: number | null
    probability_real?: number | null
    uncertainty?: number | null
    is_ood?: boolean
    ood_status?: string | null
    model_agreement?: number | null
    user_explanation?: string | null
    warnings?: VerdictWarning[]
    rgb_score?: number | null
    frequency_score?: number | null
    residual_score?: number | null
    face_score?: number | null
    temporal_score?: number | null
    evidence?: Record<string, any> | null
    forensic_flags?: string[]
    processing_time_ms?: number | null
    message?: string | null
    created_at: string
}

export interface SystemMetrics {
    total_detections: number
    real_count: number
    ai_generated_count: number
    ai_manipulated_count: number
    uncertain_count: number
    avg_processing_time_ms: number
    models_active: number
    latest_evaluations: any[]
}

export interface EvaluationReport {
    status: string
    has_trained_weights: boolean
    overall_metrics?: {
        accuracy?: number | null
        precision?: number | null
        recall?: number | null
        f1_score?: number | null
        roc_auc?: number | null
        eer?: number | null
    }
    per_generator_performance?: Array<{ generator: string; count: number; accuracy: number; f1_score: number }>
    unseen_generator_test?: {
        target_generators?: string[]
        count?: number
        accuracy?: number | null
        f1_score?: number | null
        roc_auc?: number | null
        note?: string
    }
    robustness_benchmarks?: Record<string, number | null>
    failure_analysis?: {
        false_positives_count?: number
        false_negatives_count?: number
        false_positives_samples?: any[]
        false_negatives_samples?: any[]
    }
}

async function handleResponse<T>(r: Response, fallbackError: string): Promise<T> {
    if (!r.ok) {
        let text = ''
        try {
            text = await r.text()
            const errData = JSON.parse(text)
            text = errData.detail || errData.message || text
        } catch {
            // text already holds the raw text read from r.text()
        }
        throw new Error(text || `${fallbackError} (HTTP ${r.status})`)
    }
    return r.json()
}

export async function fetchHealth(): Promise<HealthData> {
    let r: Response
    try {
        r = await fetch(`${API_BASE_URL}/health`)
    } catch {
        throw new Error('Network error: Could not connect to backend server.')
    }
    return handleResponse<HealthData>(r, 'Failed to fetch system health')
}

export async function fetchMetrics(): Promise<SystemMetrics> {
    let r: Response
    try {
        r = await fetch(`${API_BASE_URL}/metrics`)
    } catch {
        throw new Error('Network error: Could not connect to backend server.')
    }
    return handleResponse<SystemMetrics>(r, 'Failed to fetch metrics')
}

export async function fetchModels(): Promise<ModelInfo[]> {
    let r: Response
    try {
        r = await fetch(`${API_BASE_URL}/models`)
    } catch {
        throw new Error('Network error: Could not connect to backend server.')
    }
    return handleResponse<ModelInfo[]>(r, 'Failed to fetch models')
}

export async function fetchDetections(): Promise<DetectionSummary[]> {
    let r: Response
    try {
        r = await fetch(`${API_BASE_URL}/detections`)
    } catch {
        throw new Error('Network error: Could not connect to backend server.')
    }
    return handleResponse<DetectionSummary[]>(r, 'Failed to fetch detections')
}

export async function fetchEvaluationData(): Promise<EvaluationReport> {
    let r: Response
    try {
        r = await fetch(`${API_BASE_URL}/evaluation`)
    } catch {
        throw new Error('Network error: Could not connect to backend server.')
    }
    return handleResponse<EvaluationReport>(r, 'Failed to fetch evaluation report')
}

export async function detectImage(file: File): Promise<DetectionSummary> {
    const form = new FormData()
    form.append('file', file)

    let r: Response
    try {
        r = await fetch(`${API_BASE_URL}/detect/image`, {
            method: 'POST',
            body: form,
        })
    } catch {
        throw new Error('Network error: Could not connect to backend server.')
    }
    return handleResponse<DetectionSummary>(r, 'Image detection failed')
}

export async function detectVideo(file: File): Promise<DetectionSummary> {
    const form = new FormData()
    form.append('file', file)

    let r: Response
    try {
        r = await fetch(`${API_BASE_URL}/detect/video`, {
            method: 'POST',
            body: form,
        })
    } catch {
        throw new Error('Network error: Could not connect to backend server.')
    }
    return handleResponse<DetectionSummary>(r, 'Video detection failed')
}

export function useLiveHealth() {
    const [health, setHealth] = useState<HealthData | null>(null)
    const [error, setError] = useState(false)
    useEffect(() => {
        let mounted = true
        const poll = async () => {
            try {
                const h = await fetchHealth()
                if (mounted) {
                    setHealth(h)
                    setError(false)
                }
            } catch {
                if (mounted) setError(true)
            }
        }
        poll()
        const id = setInterval(poll, 15000)
        return () => {
            mounted = false
            clearInterval(id)
        }
    }, [])
    return { health, error }
}

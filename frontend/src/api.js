const API_BASE = '/api'

export async function api(path, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json', ...opts.headers },
        ...opts,
    })
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Request failed')
    }
    return res
}

export async function apiJSON(path, opts) {
    return (await api(path, opts)).json()
}

export function downloadUrl(jobId) {
    return `${API_BASE}/jobs/${jobId}/download`
}

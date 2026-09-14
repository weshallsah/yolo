import type { IdentifyResult } from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:9000'

export class IdentificationError extends Error {}

export async function identifyImage(file: File): Promise<IdentifyResult> {
  const formData = new FormData()
  formData.append('image', file)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/identify`, {
      method: 'POST',
      body: formData,
    })
  } catch {
    throw new IdentificationError('Could not reach the identification service. Is the backend running?')
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new IdentificationError(body?.detail ?? 'Could not identify a product in this image.')
  }

  return (await response.json()) as IdentifyResult
}

export async function identifyImageManual(file: File, label: string): Promise<IdentifyResult> {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('label', label)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/identify/manual`, {
      method: 'POST',
      body: formData,
    })
  } catch {
    throw new IdentificationError('Could not reach the identification service. Is the backend running?')
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new IdentificationError(body?.detail ?? 'Could not look up that item.')
  }

  return (await response.json()) as IdentifyResult
}

import type { DetectionResponse } from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:9000'

export class DetectionError extends Error {}

export async function detectImage(file: File): Promise<DetectionResponse> {
  const formData = new FormData()
  formData.append('image', file)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/detect`, {
      method: 'POST',
      body: formData,
    })
  } catch {
    throw new DetectionError('Could not reach the detection service. Is the backend running?')
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new DetectionError(body?.detail ?? 'Could not detect objects in this image.')
  }

  return (await response.json()) as DetectionResponse
}

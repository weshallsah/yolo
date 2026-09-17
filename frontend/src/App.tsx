import { useEffect, useRef, useState } from 'react'
import CameraCapture from './components/CameraCapture'
import DetectionResult from './components/DetectionResult'
import ErrorState from './components/ErrorState'
import LoadingState from './components/LoadingState'
import UploadZone from './components/UploadZone'
import { detectImage } from './lib/api'
import type { DetectionResponse } from './lib/types'

type Stage = 'idle' | 'camera' | 'analyzing' | 'result' | 'error'

function App() {
  const [stage, setStage] = useState<Stage>('idle')
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const objectUrlRef = useRef<string | null>(null)

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    }
  }, [])

  async function handleFile(file: File) {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    const url = URL.createObjectURL(file)
    objectUrlRef.current = url
    setImageUrl(url)
    setStage('analyzing')

    try {
      const data = await detectImage(file)
      setResult(data)
      setStage('result')
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Something went wrong.')
      setStage('error')
    }
  }

  function handleReset() {
    setResult(null)
    setErrorMessage(null)
    setStage('idle')
  }

  return (
    <div className="min-h-svh bg-slate-50 dark:bg-slate-950">
      <div className="mx-auto flex min-h-svh w-full max-w-2xl flex-col px-4 py-10 sm:py-16">
        <header className="mb-8 text-center">
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 sm:text-3xl">
            Object Detector
          </h1>
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
            Snap or upload a photo and see exactly what the YOLO model detects.
          </p>
        </header>

        <main className="flex flex-1 flex-col items-center justify-center rounded-3xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 sm:p-8">
          {stage === 'idle' && (
            <UploadZone onFileSelected={handleFile} onOpenCamera={() => setStage('camera')} error={null} />
          )}
          {stage === 'camera' && (
            <CameraCapture onCapture={handleFile} onClose={() => setStage('idle')} />
          )}
          {stage === 'analyzing' && imageUrl && <LoadingState imageUrl={imageUrl} />}
          {stage === 'result' && result && imageUrl && (
            <DetectionResult result={result} imageUrl={imageUrl} onReset={handleReset} />
          )}
          {stage === 'error' && errorMessage && (
            <ErrorState message={errorMessage} onRetry={handleReset} />
          )}
        </main>

        <footer className="mt-8 text-center text-xs text-slate-400">
          Detections are produced directly by the YOLO model running on the backend.
        </footer>
      </div>
    </div>
  )
}

export default App

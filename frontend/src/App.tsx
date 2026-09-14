import { useEffect, useRef, useState } from 'react'
import CameraCapture from './components/CameraCapture'
import ErrorState from './components/ErrorState'
import LoadingState from './components/LoadingState'
import ResultCard from './components/ResultCard'
import UploadZone from './components/UploadZone'
import { identifyImage, identifyImageManual } from './lib/api'
import type { IdentifyResult } from './lib/types'

type Stage = 'idle' | 'camera' | 'analyzing' | 'result' | 'error'

function App() {
  const [stage, setStage] = useState<Stage>('idle')
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [result, setResult] = useState<IdentifyResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
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
    setPendingFile(file)
    setStage('analyzing')

    try {
      const data = await identifyImage(file)
      setResult(data)
      setStage('result')
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Something went wrong.')
      setStage('error')
    }
  }

  async function handleManualIdentify(label: string) {
    if (!pendingFile) return
    setStage('analyzing')

    try {
      const data = await identifyImageManual(pendingFile, label)
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
    setPendingFile(null)
    setStage('idle')
  }

  return (
    <div className="min-h-svh bg-slate-50 dark:bg-slate-950">
      <div className="mx-auto flex min-h-svh w-full max-w-2xl flex-col px-4 py-10 sm:py-16">
        <header className="mb-8 text-center">
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 sm:text-3xl">
            What is this, and is it worth it?
          </h1>
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
            Snap or upload a photo to identify a product, compare prices, and check if it's a fair,
            trustworthy buy.
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
            <ResultCard result={result} imageUrl={imageUrl} onReset={handleReset} />
          )}
          {stage === 'error' && errorMessage && (
            <ErrorState message={errorMessage} onRetry={handleReset} onManualIdentify={handleManualIdentify} />
          )}
        </main>

        <footer className="mt-8 text-center text-xs text-slate-400">
          Prices are pulled live from real marketplace listings and may change between searches.
        </footer>
      </div>
    </div>
  )
}

export default App

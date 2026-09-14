import { useState } from 'react'

export default function ErrorState({
  message,
  onRetry,
  onManualIdentify,
}: {
  message: string
  onRetry: () => void
  onManualIdentify: (label: string) => void
}) {
  const [label, setLabel] = useState('')

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const trimmed = label.trim()
    if (!trimmed) return
    onManualIdentify(trimmed)
  }

  return (
    <div className="flex w-full flex-col items-center gap-4 rounded-2xl border border-rose-200 bg-rose-50 px-6 py-10 text-center dark:border-rose-500/30 dark:bg-rose-500/10">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-rose-100 text-2xl dark:bg-rose-500/20">
        ⚠️
      </div>
      <div>
        <p className="font-medium text-rose-700 dark:text-rose-300">We couldn't identify this item</p>
        <p className="mt-1 text-sm text-rose-600/80 dark:text-rose-400/80">{message}</p>
      </div>

      <form onSubmit={handleSubmit} className="flex w-full max-w-sm flex-col gap-2 sm:flex-row">
        <input
          type="text"
          value={label}
          onChange={(event) => setLabel(event.target.value)}
          placeholder="What is this? e.g. Motorcycle helmet"
          className="flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800 shadow-sm focus:border-indigo-400 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
        />
        <button
          type="submit"
          disabled={!label.trim()}
          className="rounded-xl bg-indigo-600 px-4 py-2.5 font-medium text-white shadow-sm transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Look it up
        </button>
      </form>

      <button
        type="button"
        onClick={onRetry}
        className="text-sm font-medium text-rose-700 underline-offset-2 hover:underline dark:text-rose-300"
      >
        Try another photo instead
      </button>
    </div>
  )
}

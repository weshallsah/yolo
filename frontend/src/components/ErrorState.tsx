export default function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex w-full flex-col items-center gap-4 rounded-2xl border border-rose-200 bg-rose-50 px-6 py-10 text-center dark:border-rose-500/30 dark:bg-rose-500/10">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-rose-100 text-2xl dark:bg-rose-500/20">
        ⚠️
      </div>
      <div>
        <p className="font-medium text-rose-700 dark:text-rose-300">We couldn't identify this item</p>
        <p className="mt-1 text-sm text-rose-600/80 dark:text-rose-400/80">{message}</p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="rounded-xl bg-rose-600 px-5 py-2.5 font-medium text-white shadow-sm transition-colors hover:bg-rose-700"
      >
        Try another photo
      </button>
    </div>
  )
}

export default function LoadingState({ imageUrl }: { imageUrl: string }) {
  return (
    <div className="flex w-full flex-col items-center gap-6 py-8">
      <div className="relative w-full max-w-sm overflow-hidden rounded-2xl">
        <img src={imageUrl} alt="Uploaded product" className="w-full object-cover" />
        <div className="scan-line absolute inset-x-0 h-1 bg-indigo-400/80 shadow-[0_0_16px_4px_rgba(99,102,241,0.6)]" />
        <div className="absolute inset-0 bg-slate-900/10" />
      </div>
      <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
        <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-500 [animation-delay:-0.3s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-500 [animation-delay:-0.15s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-500" />
        <span className="ml-2 text-sm font-medium">Identifying your item…</span>
      </div>
    </div>
  )
}

import type { IdentifyResult } from '../lib/types'
import PriceTable from './PriceTable'
import TrustBadge from './TrustBadge'

export default function ResultCard({
  result,
  imageUrl,
  onReset,
}: {
  result: IdentifyResult
  imageUrl: string
  onReset: () => void
}) {
  return (
    <div className="w-full">
      <div className="grid gap-6 sm:grid-cols-[200px_1fr]">
        <img
          src={imageUrl}
          alt={result.title}
          className="h-48 w-full rounded-xl object-cover sm:h-full"
        />

        <div className="flex flex-col gap-3 text-left">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-indigo-500 dark:text-indigo-400">
              {result.category} · {Math.round(result.confidence * 100)}% match
            </p>
            <h2 className="mt-0.5 text-xl font-semibold text-slate-900 dark:text-slate-100">
              {result.title}
            </h2>
          </div>
          <p className="text-sm text-slate-500 dark:text-slate-400">{result.description}</p>
          <TrustBadge level={result.trust.level} reason={result.trust.reason} />
        </div>
      </div>

      <div className="mt-6 text-left">
        <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-300">
          Price comparison
        </h3>
        <PriceTable listings={result.priceListings} />
      </div>

      <div className="mt-6 rounded-xl bg-indigo-50 p-4 text-left dark:bg-indigo-500/10">
        <h3 className="mb-1 text-sm font-semibold text-indigo-900 dark:text-indigo-300">
          Recommendation
        </h3>
        <p className="text-sm text-indigo-950/80 dark:text-indigo-200/80">{result.recommendation}</p>
      </div>

      <button
        type="button"
        onClick={onReset}
        className="mt-6 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
      >
        Search another image
      </button>
    </div>
  )
}

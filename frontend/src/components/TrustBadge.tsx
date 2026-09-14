import type { TrustLevel } from '../lib/types'

const STYLES: Record<TrustLevel, { label: string; classes: string; icon: string }> = {
  verified: {
    label: 'Verified',
    classes: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20 dark:bg-emerald-500/10 dark:text-emerald-400 dark:ring-emerald-500/30',
    icon: '✓',
  },
  caution: {
    label: 'Caution',
    classes: 'bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-500/10 dark:text-amber-400 dark:ring-amber-500/30',
    icon: '!',
  },
  unverified: {
    label: 'Unverified',
    classes: 'bg-rose-50 text-rose-700 ring-rose-600/20 dark:bg-rose-500/10 dark:text-rose-400 dark:ring-rose-500/30',
    icon: '×',
  },
}

export default function TrustBadge({ level, reason }: { level: TrustLevel; reason: string }) {
  const style = STYLES[level]

  return (
    <div className="flex items-start gap-2.5">
      <span
        className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${style.classes}`}
      >
        <span aria-hidden="true">{style.icon}</span>
        {style.label}
      </span>
      <p className="pt-0.5 text-sm text-slate-500 dark:text-slate-400">{reason}</p>
    </div>
  )
}

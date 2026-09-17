import type { DetectionResponse } from '../lib/types'

const BOX_COLORS = ['#6366f1', '#ec4899', '#10b981', '#f59e0b', '#0ea5e9', '#ef4444']

export default function DetectionResult({
  result,
  imageUrl,
  onReset,
}: {
  result: DetectionResponse
  imageUrl: string
  onReset: () => void
}) {
  const { detections, imageWidth, imageHeight } = result

  return (
    <div className="w-full">
      <div
        className="relative mx-auto w-full max-w-md overflow-hidden rounded-2xl bg-slate-100 dark:bg-slate-800"
        style={{ aspectRatio: `${imageWidth} / ${imageHeight}` }}
      >
        <img src={imageUrl} alt="Detected scene" className="absolute inset-0 h-full w-full object-contain" />
        {detections.map((detection, index) => {
          const color = BOX_COLORS[index % BOX_COLORS.length]
          const left = (detection.box.x1 / imageWidth) * 100
          const top = (detection.box.y1 / imageHeight) * 100
          const width = ((detection.box.x2 - detection.box.x1) / imageWidth) * 100
          const height = ((detection.box.y2 - detection.box.y1) / imageHeight) * 100

          return (
            <div
              key={index}
              className="absolute border-2"
              style={{
                left: `${left}%`,
                top: `${top}%`,
                width: `${width}%`,
                height: `${height}%`,
                borderColor: color,
              }}
            >
              <span
                className="absolute -top-6 left-0 whitespace-nowrap rounded px-1.5 py-0.5 text-xs font-medium text-white"
                style={{ backgroundColor: color }}
              >
                {detection.label} {Math.round(detection.confidence * 100)}%
              </span>
            </div>
          )
        })}
      </div>

      <div className="mt-6 text-left">
        <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-300">
          {detections.length > 0
            ? `${detections.length} object${detections.length === 1 ? '' : 's'} detected`
            : 'No objects detected'}
        </h3>
        {detections.length > 0 && (
          <ul className="flex flex-col gap-2">
            {detections.map((detection, index) => (
              <li
                key={index}
                className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm dark:border-slate-800 dark:bg-slate-900"
              >
                <span className="flex items-center gap-2 font-medium text-slate-800 dark:text-slate-200">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: BOX_COLORS[index % BOX_COLORS.length] }}
                  />
                  {detection.label}
                </span>
                <span className="text-slate-500 dark:text-slate-400">
                  {Math.round(detection.confidence * 100)}%
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <button
        type="button"
        onClick={onReset}
        className="mt-6 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
      >
        Detect another photo
      </button>
    </div>
  )
}

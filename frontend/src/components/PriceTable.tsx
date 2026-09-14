import { useMemo, useState } from 'react'
import type { PriceListing } from '../lib/types'

function formatPrice(price: number, currency: string) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(price)
}

type SortOrder = 'default' | 'asc' | 'desc'

export default function PriceTable({ listings }: { listings: PriceListing[] }) {
  const [sortOrder, setSortOrder] = useState<SortOrder>('default')

  const sortedListings = useMemo(() => {
    if (sortOrder === 'default') return listings
    const sorted = [...listings].sort((a, b) => (sortOrder === 'asc' ? a.price - b.price : b.price - a.price))
    return sorted
  }, [listings, sortOrder])

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-end gap-2 border-b border-slate-200 bg-slate-50 px-4 py-2 dark:border-slate-700 dark:bg-slate-800/60">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Sort by price:</span>
        <button
          type="button"
          onClick={() => setSortOrder(sortOrder === 'asc' ? 'default' : 'asc')}
          className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
            sortOrder === 'asc'
              ? 'bg-indigo-600 text-white'
              : 'bg-white text-slate-600 hover:bg-slate-100 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
          }`}
        >
          Low to High
        </button>
        <button
          type="button"
          onClick={() => setSortOrder(sortOrder === 'desc' ? 'default' : 'desc')}
          className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
            sortOrder === 'desc'
              ? 'bg-indigo-600 text-white'
              : 'bg-white text-slate-600 hover:bg-slate-100 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
          }`}
        >
          High to Low
        </button>
      </div>
      <table className="w-full text-left text-sm">
        <thead className="bg-slate-50 text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
          <tr>
            <th className="px-4 py-2.5 font-medium">Photo</th>
            <th className="px-4 py-2.5 font-medium">Source</th>
            <th className="px-4 py-2.5 font-medium">Condition</th>
            <th className="px-4 py-2.5 text-right font-medium">Price</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          {sortedListings.map((listing, index) => (
            <tr
              key={`${listing.source}-${listing.url}-${index}`}
              className={listing.isBestValue ? 'bg-emerald-50/60 dark:bg-emerald-500/5' : undefined}
            >
              <td className="px-4 py-2.5">
                <a href={listing.url} target="_blank" rel="noopener noreferrer">
                  {listing.thumbnail ? (
                    <img
                      src={listing.thumbnail}
                      alt={listing.source}
                      className="h-10 w-10 rounded-md object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <div className="h-10 w-10 rounded-md bg-slate-100 dark:bg-slate-800" />
                  )}
                </a>
              </td>
              <td className="px-4 py-2.5 font-medium text-slate-800 dark:text-slate-200">
                <a href={listing.url} target="_blank" rel="noopener noreferrer" className="hover:underline">
                  {listing.source}
                </a>
              </td>
              <td className="px-4 py-2.5 text-slate-500 dark:text-slate-400">{listing.condition}</td>
              <td className="px-4 py-2.5 text-right font-semibold text-slate-800 dark:text-slate-200">
                {formatPrice(listing.price, listing.currency)}
                {listing.isBestValue && (
                  <span className="ml-2 rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400">
                    Best value
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

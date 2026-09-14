export type TrustLevel = 'verified' | 'caution' | 'unverified'

export interface PriceListing {
  source: string
  price: number
  currency: string
  condition: string
  url: string
  thumbnail?: string | null
  isBestValue?: boolean
}

export interface IdentifyResult {
  id: string
  title: string
  category: string
  description: string
  confidence: number
  trust: {
    level: TrustLevel
    reason: string
  }
  priceListings: PriceListing[]
  averagePrice: number
  recommendation: string
}

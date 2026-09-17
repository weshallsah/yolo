export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Detection {
  label: string
  confidence: number
  box: BoundingBox
}

export interface DetectionResponse {
  detections: Detection[]
  imageWidth: number
  imageHeight: number
}

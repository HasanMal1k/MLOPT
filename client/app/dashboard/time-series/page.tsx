import { Suspense } from 'react'
import TimeSeriesCleaning from './time-series'

export default function TimeSeriesPage() {
  return (
    <Suspense fallback={<div>Loading time series cleaning tools...</div>}>
      <TimeSeriesCleaning />
    </Suspense>
  )
}
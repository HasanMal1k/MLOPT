// app/dashboard/feature-engineering/page.tsx
import { Suspense } from 'react'
import FeatureEngineering from './FeatureEngineering'

export default function Page() {
  return (
    <Suspense fallback={<div>Loading feature engineering...</div>}>
      <FeatureEngineering />
    </Suspense>
  )
}
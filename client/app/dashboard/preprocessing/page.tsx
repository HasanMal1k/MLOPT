import { Suspense } from 'react'
import CustomPreprocessing from './preprocessing'
export default function Page() {
  return (
    <Suspense fallback={<div>Loading feature engineering...</div>}>
      <CustomPreprocessing />
    </Suspense>
  )
}
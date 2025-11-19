-- Add metadata columns to files table for quality scores and recommendations

-- Add quality score (0-100) based on data completeness, size, and feature quality
ALTER TABLE public.files 
ADD COLUMN IF NOT EXISTS quality_score numeric CHECK (quality_score >= 0 AND quality_score <= 100);

-- Add quality rating (Excellent, Good, Fair, Poor)
ALTER TABLE public.files 
ADD COLUMN IF NOT EXISTS quality_rating text CHECK (quality_rating IN ('Excellent', 'Good', 'Fair', 'Poor'));

-- Add recommended models as JSON array
ALTER TABLE public.files 
ADD COLUMN IF NOT EXISTS recommended_models jsonb DEFAULT '[]'::jsonb;

-- Add data characteristics for model selection
ALTER TABLE public.files 
ADD COLUMN IF NOT EXISTS data_characteristics jsonb DEFAULT '{}'::jsonb;

-- Add comments
COMMENT ON COLUMN public.files.quality_score IS 'Calculated quality score (0-100) based on completeness, size, and feature quality';
COMMENT ON COLUMN public.files.quality_rating IS 'Quality rating: Excellent (90+), Good (70-89), Fair (50-69), Poor (<50)';
COMMENT ON COLUMN public.files.recommended_models IS 'Array of recommended ML algorithms based on data characteristics';
COMMENT ON COLUMN public.files.data_characteristics IS 'JSON object containing: numeric_ratio, categorical_ratio, missing_ratio, imbalance_ratio, feature_count, etc.';

-- Create index for quality-based queries
CREATE INDEX IF NOT EXISTS idx_files_quality_score ON public.files(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_files_quality_rating ON public.files(quality_rating);

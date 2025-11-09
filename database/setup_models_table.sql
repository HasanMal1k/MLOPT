-- ============================================
-- TRAINED MODELS TABLE SETUP
-- ============================================
-- This table stores trained ML models for each user
-- Users can save model pickle files and metadata
-- Supports download and management of trained models
-- ============================================

-- Create trained_models table
CREATE TABLE IF NOT EXISTS public.trained_models (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  file_id uuid,  -- Reference to the dataset used for training
  
  -- Model Information
  model_name text NOT NULL,  -- User-provided name for the model
  model_type text NOT NULL,  -- e.g., 'classification', 'regression', 'time_series'
  algorithm text NOT NULL,  -- e.g., 'RandomForest', 'XGBoost', 'LSTM'
  
  -- Model Performance
  metrics jsonb NOT NULL DEFAULT '{}'::jsonb,  -- Accuracy, F1, RMSE, etc.
  training_config jsonb NOT NULL DEFAULT '{}'::jsonb,  -- Hyperparameters used
  
  -- File Storage
  model_file_path text NOT NULL,  -- Supabase Storage path to pickle file
  model_file_size bigint NOT NULL,  -- Size in bytes
  model_file_url text,  -- Signed URL for download (temporary)
  
  -- Model Details
  feature_columns text[] NOT NULL DEFAULT '{}'::text[],  -- Features used
  target_column text,  -- Target variable
  preprocessing_steps jsonb DEFAULT '{}'::jsonb,  -- Preprocessing applied
  
  -- Training Info
  training_time_seconds numeric,  -- How long training took
  training_samples integer,  -- Number of samples used
  test_samples integer,  -- Number of test samples
  
  -- Status & Metadata
  status text NOT NULL DEFAULT 'ready'::text,  -- 'training', 'ready', 'failed', 'deleted'
  description text,  -- User notes about the model
  tags text[] DEFAULT '{}'::text[],  -- User-defined tags
  
  -- Timestamps
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  last_downloaded_at timestamp with time zone,
  
  -- Constraints
  CONSTRAINT trained_models_pkey PRIMARY KEY (id),
  CONSTRAINT trained_models_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE,
  CONSTRAINT trained_models_file_id_fkey FOREIGN KEY (file_id) REFERENCES public.files(id) ON DELETE SET NULL,
  CONSTRAINT trained_models_status_check CHECK (status IN ('training', 'ready', 'failed', 'deleted'))
);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

-- Index on user_id for fast user queries
CREATE INDEX IF NOT EXISTS idx_trained_models_user_id 
ON public.trained_models(user_id);

-- Index on file_id for dataset relationship
CREATE INDEX IF NOT EXISTS idx_trained_models_file_id 
ON public.trained_models(file_id);

-- Index on status for filtering
CREATE INDEX IF NOT EXISTS idx_trained_models_status 
ON public.trained_models(status);

-- Index on created_at for sorting by date
CREATE INDEX IF NOT EXISTS idx_trained_models_created_at 
ON public.trained_models(created_at DESC);

-- Composite index for user + status queries
CREATE INDEX IF NOT EXISTS idx_trained_models_user_status 
ON public.trained_models(user_id, status);

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS
ALTER TABLE public.trained_models ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can view their own models" ON public.trained_models;
DROP POLICY IF EXISTS "Users can insert their own models" ON public.trained_models;
DROP POLICY IF EXISTS "Users can update their own models" ON public.trained_models;
DROP POLICY IF EXISTS "Users can delete their own models" ON public.trained_models;

-- Policy: Users can only see their own models
CREATE POLICY "Users can view their own models"
ON public.trained_models
FOR SELECT
USING (auth.uid() = user_id);

-- Policy: Users can only create models for themselves
CREATE POLICY "Users can insert their own models"
ON public.trained_models
FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Policy: Users can only update their own models
CREATE POLICY "Users can update their own models"
ON public.trained_models
FOR UPDATE
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

-- Policy: Users can only delete their own models
CREATE POLICY "Users can delete their own models"
ON public.trained_models
FOR DELETE
USING (auth.uid() = user_id);

-- ============================================
-- AUTOMATIC TIMESTAMP UPDATE TRIGGER
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_trained_models_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop trigger if exists
DROP TRIGGER IF EXISTS trigger_update_trained_models_timestamp ON public.trained_models;

-- Create trigger
CREATE TRIGGER trigger_update_trained_models_timestamp
BEFORE UPDATE ON public.trained_models
FOR EACH ROW
EXECUTE FUNCTION update_trained_models_updated_at();

-- ============================================
-- STORAGE BUCKET FOR MODEL FILES
-- ============================================

-- Create storage bucket for model pickle files
-- Run this in Supabase Dashboard -> Storage or via SQL
INSERT INTO storage.buckets (id, name, public)
VALUES ('model-files', 'model-files', false)
ON CONFLICT (id) DO NOTHING;

-- Drop existing storage policies if they exist
DROP POLICY IF EXISTS "Users can upload their own model files" ON storage.objects;
DROP POLICY IF EXISTS "Users can download their own model files" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete their own model files" ON storage.objects;

-- Storage policies for model files bucket
-- Users can upload their own model files
CREATE POLICY "Users can upload their own model files"
ON storage.objects FOR INSERT
WITH CHECK (
  bucket_id = 'model-files' 
  AND auth.uid()::text = (storage.foldername(name))[1]
);

-- Users can download their own model files
CREATE POLICY "Users can download their own model files"
ON storage.objects FOR SELECT
USING (
  bucket_id = 'model-files' 
  AND auth.uid()::text = (storage.foldername(name))[1]
);

-- Users can delete their own model files
CREATE POLICY "Users can delete their own model files"
ON storage.objects FOR DELETE
USING (
  bucket_id = 'model-files' 
  AND auth.uid()::text = (storage.foldername(name))[1]
);

-- ============================================
-- HELPER VIEWS (OPTIONAL)
-- ============================================

-- View for model statistics per user
CREATE OR REPLACE VIEW user_model_stats AS
SELECT 
  user_id,
  COUNT(*) as total_models,
  COUNT(*) FILTER (WHERE status = 'ready') as ready_models,
  COUNT(*) FILTER (WHERE status = 'training') as training_models,
  COUNT(*) FILTER (WHERE status = 'failed') as failed_models,
  SUM(model_file_size) as total_storage_bytes,
  MAX(created_at) as last_model_created
FROM public.trained_models
GROUP BY user_id;

-- Grant access to view
GRANT SELECT ON user_model_stats TO authenticated;

-- ============================================
-- SAMPLE QUERIES (FOR TESTING)
-- ============================================

-- Query 1: Get all models for current user
-- SELECT * FROM trained_models WHERE user_id = auth.uid() ORDER BY created_at DESC;

-- Query 2: Get model with metrics
-- SELECT id, model_name, algorithm, metrics, created_at 
-- FROM trained_models 
-- WHERE user_id = auth.uid() AND status = 'ready';

-- Query 3: Get models by type
-- SELECT * FROM trained_models 
-- WHERE user_id = auth.uid() AND model_type = 'classification';

-- Query 4: Get user's model statistics
-- SELECT * FROM user_model_stats WHERE user_id = auth.uid();

-- ============================================
-- SUCCESS MESSAGE
-- ============================================

-- If you see this, the setup was successful!
DO $$ 
BEGIN 
  RAISE NOTICE '✅ Trained models table created successfully!';
  RAISE NOTICE '✅ Indexes created for performance';
  RAISE NOTICE '✅ Row Level Security enabled';
  RAISE NOTICE '✅ Storage bucket configured';
  RAISE NOTICE '✅ All policies applied';
  RAISE NOTICE '';
  RAISE NOTICE '🎉 Ready to save trained models!';
  RAISE NOTICE '';
  RAISE NOTICE '📝 Table: public.trained_models';
  RAISE NOTICE '📦 Storage: model-files bucket';
  RAISE NOTICE '🔒 RLS: Enabled (users can only access their own models)';
END $$;

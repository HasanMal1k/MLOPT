-- ============================================
-- FIX: Update user_model_stats view to exclude deleted models
-- ============================================
-- This fixes the bug where deleted models are still counted in stats
-- Run this in Supabase SQL Editor
-- ============================================

-- Drop and recreate the view with proper filtering
DROP VIEW IF EXISTS user_model_stats;

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
WHERE status != 'deleted'  -- ✅ EXCLUDE DELETED MODELS
GROUP BY user_id;

-- Grant access to authenticated users
GRANT SELECT ON user_model_stats TO authenticated;

-- Success message
DO $$ 
BEGIN 
  RAISE NOTICE '✅ user_model_stats view updated to exclude deleted models';
END $$;

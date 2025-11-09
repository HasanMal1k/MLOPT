-- ============================================
-- AZURE DEPLOYMENTS TABLE SETUP
-- ============================================
-- This table stores Azure ML deployment information for each user
-- Users can track their deployed models and endpoints
-- ============================================

-- Create deployments table
CREATE TABLE IF NOT EXISTS public.deployments (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  model_id uuid NOT NULL,  -- Reference to trained_models table
  
  -- Deployment Information
  deployment_name text NOT NULL,
  endpoint_name text NOT NULL,
  azure_model_name text NOT NULL,
  azure_model_version text,
  
  -- Azure Resources
  scoring_uri text,
  swagger_uri text,
  instance_type text NOT NULL DEFAULT 'Standard_DS1_v2'::text,
  instance_count integer NOT NULL DEFAULT 1,
  
  -- Status
  status text NOT NULL DEFAULT 'deploying'::text,  -- 'deploying', 'active', 'failed', 'deleted'
  error_message text,
  
  -- Metadata
  description text,
  deployment_config jsonb DEFAULT '{}'::jsonb,
  
  -- Timestamps
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  deployed_at timestamp with time zone,
  deleted_at timestamp with time zone,
  
  -- Constraints
  CONSTRAINT deployments_pkey PRIMARY KEY (id),
  CONSTRAINT deployments_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE,
  CONSTRAINT deployments_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.trained_models(id) ON DELETE CASCADE,
  CONSTRAINT deployments_status_check CHECK (status IN ('deploying', 'active', 'failed', 'deleted'))
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_deployments_user_id ON public.deployments(user_id);
CREATE INDEX IF NOT EXISTS idx_deployments_model_id ON public.deployments(model_id);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON public.deployments(status);
CREATE INDEX IF NOT EXISTS idx_deployments_endpoint_name ON public.deployments(endpoint_name);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

ALTER TABLE public.deployments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own deployments" ON public.deployments;
DROP POLICY IF EXISTS "Users can insert their own deployments" ON public.deployments;
DROP POLICY IF EXISTS "Users can update their own deployments" ON public.deployments;
DROP POLICY IF EXISTS "Users can delete their own deployments" ON public.deployments;

CREATE POLICY "Users can view their own deployments"
ON public.deployments FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own deployments"
ON public.deployments FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own deployments"
ON public.deployments FOR UPDATE
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own deployments"
ON public.deployments FOR DELETE
USING (auth.uid() = user_id);

-- ============================================
-- AUTOMATIC TIMESTAMP UPDATE TRIGGER
-- ============================================

CREATE OR REPLACE FUNCTION update_deployments_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_deployments_timestamp ON public.deployments;

CREATE TRIGGER trigger_update_deployments_timestamp
BEFORE UPDATE ON public.deployments
FOR EACH ROW
EXECUTE FUNCTION update_deployments_updated_at();

-- ============================================
-- HELPER VIEWS
-- ============================================

CREATE OR REPLACE VIEW user_deployment_stats AS
SELECT 
  user_id,
  COUNT(*) as total_deployments,
  COUNT(*) FILTER (WHERE status = 'active') as active_deployments,
  COUNT(*) FILTER (WHERE status = 'deploying') as deploying_count,
  COUNT(*) FILTER (WHERE status = 'failed') as failed_deployments,
  MAX(deployed_at) as last_deployment_date
FROM public.deployments
WHERE status != 'deleted'
GROUP BY user_id;

GRANT SELECT ON user_deployment_stats TO authenticated;

-- ============================================
-- SUCCESS MESSAGE
-- ============================================

DO $$ 
BEGIN 
  RAISE NOTICE '✅ Deployments table created successfully!';
  RAISE NOTICE '✅ RLS policies applied';
  RAISE NOTICE '✅ Indexes created';
  RAISE NOTICE '🎉 Ready to deploy models to Azure!';
END $$;

-- Migration script to add 'name' field to existing files table
-- This field stores the unique display name that users provide during upload

-- Add the name column
ALTER TABLE public.files ADD COLUMN IF NOT EXISTS name text;

-- Populate name column with unique values by appending row number to duplicates
WITH numbered_rows AS (
  SELECT 
    id,
    regexp_replace(original_filename, '\.(csv|xlsx|xls)$', '', 'i') as base_name,
    ROW_NUMBER() OVER (PARTITION BY regexp_replace(original_filename, '\.(csv|xlsx|xls)$', '', 'i') ORDER BY upload_date) as rn
  FROM public.files
  WHERE name IS NULL
)
UPDATE public.files 
SET name = CASE 
  WHEN numbered_rows.rn = 1 THEN numbered_rows.base_name
  ELSE numbered_rows.base_name || '_' || numbered_rows.rn
END
FROM numbered_rows
WHERE public.files.id = numbered_rows.id;

-- Make name column NOT NULL
ALTER TABLE public.files ALTER COLUMN name SET NOT NULL;

-- Add unique constraint on name
ALTER TABLE public.files ADD CONSTRAINT files_name_unique UNIQUE (name);

-- Create index on name field for faster uniqueness checks
CREATE INDEX IF NOT EXISTS idx_files_name ON public.files(name);

-- Create index on user_id for faster user-specific queries (if not exists)
CREATE INDEX IF NOT EXISTS idx_files_user_id ON public.files(user_id);

-- Add comment to explain the name field
COMMENT ON COLUMN public.files.name IS 'User-provided custom display name for the file. Must be unique across all users.';

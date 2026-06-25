ALTER TABLE jobs
ADD COLUMN IF NOT EXISTS time_costs jsonb NOT NULL DEFAULT '[]'::jsonb;

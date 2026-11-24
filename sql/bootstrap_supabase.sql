-- Reference bootstrap SQL for a NEW FXB journal table in another project.
-- The connected project's table was provisioned through Supabase migrations.
-- Execute only in the intended project; transaction rolls back if the table exists.
BEGIN;
CREATE TABLE public.fxb_events (
    id text PRIMARY KEY,
    run_id text NOT NULL,
    symbol text NOT NULL,
    kind text NOT NULL CHECK (kind ~ '^[a-z][a-z0-9_]{0,63}$'),
    occurred_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    received_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT fxb_chronology CHECK (available_at >= occurred_at)
);
CREATE INDEX fxb_events_recall_idx ON public.fxb_events (run_id, symbol, kind, available_at DESC);
ALTER TABLE public.fxb_events ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.fxb_events FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, INSERT, UPDATE ON public.fxb_events TO service_role;
COMMENT ON TABLE public.fxb_events IS 'Private FXB execution and settled-reward journal. No browser access.';
COMMIT;

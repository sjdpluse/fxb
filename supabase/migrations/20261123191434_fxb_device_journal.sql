-- Existing journal stays private. Device credentials can only append FXB events.
ALTER TABLE public.fxb_events ALTER COLUMN id TYPE text USING id::text;
ALTER TABLE public.fxb_events DROP CONSTRAINT fxb_events_kind_check;
ALTER TABLE public.fxb_events ADD CONSTRAINT fxb_events_kind_check
  CHECK (kind ~ '^[a-z][a-z0-9_]{0,63}$');
CREATE TABLE public.fxb_devices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  label text NOT NULL,
  token_hash text NOT NULL UNIQUE CHECK (token_hash ~ '^[0-9a-f]{64}$'),
  revoked boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at timestamptz
);
ALTER TABLE public.fxb_devices ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.fxb_devices FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON public.fxb_devices TO service_role;
ALTER TABLE public.fxb_events ADD COLUMN device_id uuid REFERENCES public.fxb_devices(id);
CREATE INDEX fxb_events_device_time_idx ON public.fxb_events(device_id, available_at DESC);
COMMENT ON TABLE public.fxb_events IS 'Private FXB execution and settled-reward journal. No browser access.';

CREATE FUNCTION public.fxb_ingest(p_token_hash text, p_events jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path = '' AS $$
DECLARE device uuid; event record; old public.fxb_events; n integer := 0;
BEGIN
  SELECT id INTO device FROM public.fxb_devices WHERE token_hash = p_token_hash AND NOT revoked;
  IF device IS NULL THEN RAISE EXCEPTION 'Invalid device credential' USING ERRCODE = '28000'; END IF;
  IF jsonb_typeof(p_events) <> 'array' OR jsonb_array_length(p_events) > 100
      OR octet_length(p_events::text) > 1048576 THEN
    RAISE EXCEPTION 'Invalid batch';
  END IF;
  FOR event IN SELECT * FROM jsonb_to_recordset(p_events) AS x(
    id text, run_id text, symbol text, kind text, occurred_at timestamptz,
    available_at timestamptz, payload jsonb)
  LOOP
    IF event.id IS NULL OR event.id !~ '^[a-zA-Z0-9:_-]{1,128}$'
        OR length(event.run_id) > 256 OR length(event.symbol) > 64
        OR event.run_id IS NULL OR event.symbol IS NULL THEN
      RAISE EXCEPTION 'Invalid event';
    END IF;
    INSERT INTO public.fxb_events(id,run_id,symbol,kind,occurred_at,available_at,payload,device_id)
      VALUES(event.id,event.run_id,event.symbol,event.kind,event.occurred_at,event.available_at,event.payload,device)
      ON CONFLICT(id) DO NOTHING;
    SELECT * INTO old FROM public.fxb_events WHERE id=event.id;
    IF old.device_id IS DISTINCT FROM device OR old.run_id IS DISTINCT FROM event.run_id
        OR old.symbol IS DISTINCT FROM event.symbol OR old.kind IS DISTINCT FROM event.kind
        OR old.occurred_at IS DISTINCT FROM event.occurred_at OR old.available_at IS DISTINCT FROM event.available_at
        OR old.payload IS DISTINCT FROM event.payload THEN
      RAISE EXCEPTION 'Event identity conflict';
    END IF;
    n := n+1;
  END LOOP;
  UPDATE public.fxb_devices SET last_seen_at=now() WHERE id=device;
  RETURN jsonb_build_object('accepted',n,'device_id',device);
END;
$$;
REVOKE ALL ON FUNCTION public.fxb_ingest(text,jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.fxb_ingest(text,jsonb) TO service_role;

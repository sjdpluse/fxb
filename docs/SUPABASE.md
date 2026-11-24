# Private FXB journal connection

Project URL: `https://cvernmhpnsgdtegvblyc.supabase.co`. A dashboard link is now
normalized to the HTTPS project root. No MT5 credentials are sent to Supabase.

1. Update FXB and open **Memory / حافظه و گزارش**.
2. Import the private `FXB-Connection.json` supplied to the project owner.
3. FXB verifies the credential with the deployed journal before saving it.
4. Look for **connected**, the acknowledgment time and pending-event count.
   The outbox runs every 30 seconds while the workstation is open. An explicit
   sync is also available; connection failures retain events for retry.

Keep this file private. It is intentionally absent from Git and contains only
a scoped, revocable device credential, not a Supabase project secret. Windows
stores the local credential using current-user DPAPI. Other platforms use an
owner-only file. It is never returned by the status API or stored in browser
storage. A replacement computer must import the original private file again.

The `fxb-journal` Edge Function has its own device authentication; platform
`verify_jwt=false` does NOT make ingestion anonymous. It hashes the credential,
then calls a service-role-only SQL function. The function checks revocation,
validates bounded batches and inserts immutable events in one transaction.
Same-ID identical retries succeed; modified payloads or another device's ID
fail. RLS and grants deny anon/authenticated access to both FXB tables; no
SECURITY DEFINER function or blanket browser-write policy is used. Existing
Saraf/FXC tables and auth settings are untouched.

The old table accepted only UUID IDs and three event kinds. Runtime rewards
use SHA-256 IDs and kinds such as `scalp_reward`/`micro_decision`, so a URL fix
alone could not sync them. The migration changes IDs to text, validates kind
syntax, and adds device ownership. A 64-character connection-check event was
uploaded and queried successfully; duplicate retries, conflicting rewrites and
invalid credentials were tested against the live backend. This event is marked
`learning_sample=false`, and is not a trade or model reward.

Learning and trading remain local. Rewards include actual net broker P/L,
prediction error, updated readout weights and revision. The cloud is an audit
copy; it does not invent market explanations, train biological synapses or
automatically replace live weights from a remote record. No cloud request is
made on the order-management thread.

Deployment: migrations were created with pinned Supabase CLI 2.54.11 and applied
through the project migration API. For an existing bootstrapped journal apply
`supabase/migrations/*_fxb_device_journal.sql`; for a new project first apply
`sql/bootstrap_supabase.sql`. Deploy `supabase/functions/fxb-journal` using its
bundled `deno.json` with gateway JWT checking off and the handler's own auth on.
Provision devices privately by storing only SHA-256(token), where token is
`fxb_device_` plus 32 cryptographically random bytes encoded as hex. Never put
generated credentials or IDs into migration files. Revoke a device by setting
`fxb_devices.revoked=true` with administrative access.

Legacy `FXB_SUPABASE_URL` + backend `FXB_SUPABASE_SECRET_KEY` environment settings
remain supported if no device file exists. The UI no longer needs those keys.
`python -m fxb memory --db state/runtime.sqlite --sync` uses the same device
connection. A project secret must never be shared in chat, Git or frontend code.

// No project-wide secret leaves the Edge runtime. The workstation has a scoped
// random credential; only its SHA-256 hash is stored, with explicit revocation.
export async function handle(req: Request): Promise<Response> {
  const reply = (status: number, body: object) => Response.json(body, {
    status, headers: {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
  });
  if (req.method !== "POST") return reply(405, {error: "POST required"});
  if (req.headers.has("origin")) return reply(403, {error: "Backend only"});
  const token = req.headers.get("authorization")?.replace(/^Bearer /, "") ?? "";
  if (!/^fxb_device_[0-9a-f]{64}$/.test(token)) return reply(401, {error: "Invalid device credential"});
  if (!req.body) return reply(400, {error: "Missing batch"});
  const reader = req.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 1048576) {await reader.cancel(); return reply(413, {error: "Batch too large"});}
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {bytes.set(chunk, offset); offset += chunk.length;}
    const {events} = JSON.parse(new TextDecoder().decode(bytes));
    if (!Array.isArray(events) || events.length > 100) return reply(400, {error: "Invalid batch"});
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(token));
    const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
    const key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const res = await fetch(Deno.env.get("SUPABASE_URL")+"/rest/v1/rpc/fxb_ingest", {
      method: "POST", redirect: "error", signal: AbortSignal.timeout(12000),
      headers: {"Content-Type": "application/json", "apikey": key, "Authorization": "Bearer "+key},
      body: JSON.stringify({p_token_hash: hash, p_events: events}),
    });
    const result = await res.json();
    if (!res.ok) return reply(result.code === "28000" ? 401 : 409, {error: "Journal batch rejected"});
    return reply(200, result);
  } catch {
    return reply(400, {error: "Invalid or unavailable journal request"});
  }
}
if (import.meta.main) Deno.serve(handle);

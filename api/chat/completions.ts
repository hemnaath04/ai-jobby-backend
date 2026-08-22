// ---------------------------------------------------------------------------
// RoleReveal backend proxy (Vercel Node.js Function).
//
// Why this exists: a Chrome extension is public JS, so any key shipped in it is
// extractable. This proxy holds the real LLM keys SERVER-SIDE (env vars) and the
// extension calls this endpoint instead. It's OpenAI-compatible, so the
// extension's existing "custom" provider works unchanged — just point its base
// URL at `https://<this-deployment>/api`.
//
// Model routing: Groq FIRST — a genuinely non-reasoning, fast provider (500+
// tok/s on Llama 3.1 8B, 250+ tok/s on Llama 3.3 70B; generous free tier).
// Falls back to OpenRouter's free model catalog on any Groq failure (missing
// key, error, rate limit, timeout, empty/truncated response), with reasoning
// explicitly disabled (`reasoning: { enabled: false }`) — every current
// OpenRouter free model advertises that toggle. The client's `model` field
// (the extension always sends "auto") is ignored; the model is always chosen
// server-side per leg below.
//
// Manifest is no longer in this path: it silently overrides/strips
// per-request reasoning flags (measured: reasoning stayed on even after
// sending reasoning:{enabled:false} through it) and several of its free
// fallback models are reasoning models that burn 100+ hidden tokens before
// writing the visible answer, which is what made scoring slow in the first
// place.
//
// Route: POST /api/chat/completions   (Vercel maps this file to that path)
//
// Runs on the Node.js runtime, not Edge: Edge Functions must begin sending a
// response within 25 seconds or Vercel kills the invocation, and a
// primary-then-fallback call can occasionally take longer than that. Node.js
// Functions only have the maxDuration budget below, no first-byte deadline.
export const config = { maxDuration: 30 };

// Vercel provides process.env at runtime; declare it so we don't need @types/node.
declare const process: { env: Record<string, string | undefined> };

const CORS: Record<string, string> = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'POST, OPTIONS',
  'access-control-allow-headers': 'authorization, content-type, x-app-token, x-client-id',
  'access-control-max-age': '86400',
};

const json = (data: unknown, status = 200, extra: Record<string, string> = {}) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json', ...CORS, ...extra },
  });

function clampNum(v: unknown, min: number, max: number, dflt: number): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return dflt;
  return Math.min(max, Math.max(min, n));
}

// ── Strict, persistent rate limiting via Upstash Redis ──────────────────────
// Counters survive across invocations. Enforced ONLY when Upstash env vars
// are set — set them before going public, or there is no shared counter to
// enforce against (serverless has no shared memory).
function redisConfigured(): boolean {
  return !!(process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN);
}

async function incrWithTtl(key: string, ttlSeconds: number): Promise<number | null> {
  const url = process.env.UPSTASH_REDIS_REST_URL!;
  const tok = process.env.UPSTASH_REDIS_REST_TOKEN!;
  try {
    const r = await fetch(`${url}/incr/${encodeURIComponent(key)}`, {
      headers: { authorization: `Bearer ${tok}` },
    });
    const count = Number((await r.json()).result);
    if (count === 1) {
      await fetch(`${url}/expire/${encodeURIComponent(key)}/${ttlSeconds}`, {
        headers: { authorization: `Bearer ${tok}` },
      });
    }
    return count;
  } catch {
    return null; // never hard-fail a request because the limiter glitched
  }
}

const dayStamp = () => new Date().toISOString().slice(0, 10).replace(/-/g, '');
const secsToUtcMidnight = () => {
  const now = new Date();
  const next = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
  return Math.max(1, Math.floor((next.getTime() - now.getTime()) / 1000));
};

interface LimitVerdict {
  ok: boolean;
  error?: string;
  message?: string;
  retryAfter?: number;
}

async function checkLimits(clientId: string, ip: string): Promise<LimitVerdict> {
  if (!redisConfigured()) return { ok: true }; // can't enforce without a store
  const day = dayStamp();
  const perUser = Number(process.env.DAILY_LIMIT_PER_USER || '100');
  const perIpDay = Number(process.env.DAILY_LIMIT_PER_IP || '300');
  const perMin = Number(process.env.RATE_LIMIT_PER_MIN || '20');
  const globalDay = Number(process.env.GLOBAL_DAILY_CAP || '0'); // 0 = disabled

  // 1) Burst per IP (per minute).
  const burst = await incrWithTtl(`aj:min:${ip}:${Math.floor(Date.now() / 60000)}`, 120);
  if (burst !== null && burst > perMin)
    return { ok: false, error: 'rate_limited', message: 'Too many requests, slow down.', retryAfter: 60 };

  // 2) Daily per IP (defeats client-id rotation).
  const ipDay = await incrWithTtl(`aj:ipday:${ip}:${day}`, 90000);
  if (ipDay !== null && ipDay > perIpDay)
    return { ok: false, error: 'daily_ip_limit', message: 'Daily limit reached for this network.', retryAfter: secsToUtcMidnight() };

  // 3) Daily per user (the 100/day cap).
  const userDay = await incrWithTtl(`aj:uday:${clientId || ip}:${day}`, 90000);
  if (userDay !== null && userDay > perUser)
    return {
      ok: false,
      error: 'daily_limit_reached',
      message: `Daily limit of ${perUser} scored jobs reached. Try again tomorrow, or add your own API key in RoleReveal → Options.`,
      retryAfter: secsToUtcMidnight(),
    };

  // 4) Optional global backstop across all users.
  if (globalDay > 0) {
    const all = await incrWithTtl(`aj:gday:${day}`, 90000);
    if (all !== null && all > globalDay)
      return { ok: false, error: 'service_busy', message: 'Daily capacity reached. Please try again tomorrow.', retryAfter: secsToUtcMidnight() };
  }

  return { ok: true };
}

interface LegResult {
  status: number;
  text: string;
}

async function callLeg(
  url: string,
  key: string,
  body: Record<string, unknown>,
  timeoutMs: number,
  extraHeaders: Record<string, string> = {},
): Promise<LegResult | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${key}`, ...extraHeaders },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    return { status: res.status, text };
  } catch (e: any) {
    console.error(`[leg unreachable/timeout] ${String(e?.message || e)}`);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// A leg is only "usable" if it succeeded AND actually finished writing an
// answer. A 200 with empty content, or one cut off by finish_reason "length"
// (a reasoning model spending its whole budget on hidden thinking), isn't
// something worth returning to the extension — fall through to the next leg
// instead of surfacing "empty response" / "No JSON object found" errors.
function isUsable(leg: LegResult | null): boolean {
  if (!leg || leg.status < 200 || leg.status >= 300) return false;
  try {
    const choice = JSON.parse(leg.text)?.choices?.[0];
    const content = choice?.message?.content;
    const truncated = choice?.finish_reason === 'length' || choice?.native_finish_reason === 'length';
    const hasContent = typeof content === 'string' ? content.trim() !== '' : !!content;
    return hasContent && !truncated;
  } catch {
    return false;
  }
}

async function handler(req: Request): Promise<Response> {
  if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: CORS });
  if (req.method !== 'POST') return json({ error: 'method_not_allowed' }, 405);

  const GROQ_KEY = process.env.GROQ_API_KEY;
  const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY;
  if (!GROQ_KEY && !OPENROUTER_KEY) return json({ error: 'server_not_configured' }, 500);

  // Optional revocable app token. If APP_TOKEN is set, the extension must send a
  // matching Authorization: Bearer <token> (or x-app-token). It's still public
  // (it's in the extension), but you can rotate it to cut off abuse.
  const appToken = process.env.APP_TOKEN;
  if (appToken) {
    const sent =
      req.headers.get('x-app-token') ||
      (req.headers.get('authorization') || '').replace(/^Bearer\s+/i, '');
    if (sent !== appToken) return json({ error: 'unauthorized' }, 401);
  }

  let body: any;
  try {
    body = await req.json();
  } catch {
    return json({ error: 'bad_json' }, 400);
  }

  const messages = body?.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    return json({ error: 'messages_required' }, 400);
  }
  // Bound prompt size so nobody runs huge jobs on your key.
  if (JSON.stringify(messages).length > 140_000) {
    return json({ error: 'request_too_large' }, 413);
  }

  const ip = (req.headers.get('x-forwarded-for') || '').split(',')[0].trim() || 'anon';
  const clientId = (req.headers.get('x-client-id') || '').slice(0, 64);
  const verdict = await checkLimits(clientId, ip);
  if (!verdict.ok) {
    return json(
      { error: verdict.error, message: verdict.message },
      429,
      { 'retry-after': String(verdict.retryAfter ?? 60) },
    );
  }

  // Shared fields; model, and any provider-specific extras, are added per leg.
  const sharedBody: Record<string, unknown> = {
    messages,
    temperature: clampNum(body.temperature, 0, 2, 0.2),
    max_tokens: clampNum(body.max_tokens, 1, 4096, 900),
    ...(body.response_format ? { response_format: body.response_format } : {}),
  };

  let leg: LegResult | null = null;
  let usedProvider = 'none';

  // 1) Groq: fast, non-reasoning models. Short timeout — at 250-500+ tok/s a
  // real answer lands in low single-digit seconds, so anything slower than
  // this is worth failing over rather than waiting on.
  if (GROQ_KEY) {
    // Groq's classic Llama chat models (llama-3.1-8b-instant,
    // llama-3.3-70b-versatile) were retired from this account's catalog —
    // verified live against GET /openai/v1/models on 2026-08-22. gpt-oss-20b
    // is the smallest/fastest chat model currently available.
    const groqModel = process.env.GROQ_MODEL || 'openai/gpt-oss-20b';
    leg = await callLeg(
      'https://api.groq.com/openai/v1/chat/completions',
      GROQ_KEY,
      { ...sharedBody, model: groqModel },
      Number(process.env.GROQ_TIMEOUT_MS || '10000'),
    );
    if (leg) console.error(`[groq] status=${leg.status} usable=${isUsable(leg)} body=${leg.text.slice(0, 300)}`);
    if (isUsable(leg)) usedProvider = 'groq';
  }

  // 2) OpenRouter free catalog, reasoning explicitly disabled, as fallback.
  if (usedProvider === 'none' && OPENROUTER_KEY) {
    const orModel = process.env.OPENROUTER_MODEL || 'nvidia/nemotron-3.5-lightning:free';
    leg = await callLeg(
      'https://openrouter.ai/api/v1/chat/completions',
      OPENROUTER_KEY,
      { ...sharedBody, model: orModel, reasoning: { enabled: false } },
      Number(process.env.OPENROUTER_TIMEOUT_MS || '15000'),
      { 'http-referer': 'https://rolereveal.app', 'x-title': 'RoleReveal' },
    );
    if (leg) console.error(`[openrouter] status=${leg.status} usable=${isUsable(leg)} body=${leg.text.slice(0, 300)}`);
    if (isUsable(leg)) usedProvider = 'openrouter';
  }

  if (!leg) return json({ error: 'upstream_unreachable' }, 502);

  // Pass the OpenAI-shaped response straight through (extension reads
  // choices[0]) — including a final unusable result, so the extension's own
  // error messages (empty response / bad JSON) still surface if both legs
  // failed, rather than masking it with a generic 502.
  return new Response(leg.text, {
    status: leg.status,
    headers: { 'content-type': 'application/json', ...CORS, 'x-rr-provider': usedProvider },
  });
}

// Node.js Vercel Function using the documented `fetch` Web Standard export so
// this one handler covers every HTTP method (OPTIONS/POST/etc — see the
// method check at the top of handler()) without a classic (req, res) signature.
export default { fetch: handler };

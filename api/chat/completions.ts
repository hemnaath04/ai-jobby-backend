// ---------------------------------------------------------------------------
// RoleReveal backend proxy (Vercel Edge Function).
//
// Why this exists: a Chrome extension is public JS, so any key shipped in it is
// extractable. This proxy holds the real LLM key SERVER-SIDE (env var) and the
// extension calls this endpoint instead. It's OpenAI-compatible, so the
// extension's existing "custom" provider works unchanged — just point its base
// URL at `https://<this-deployment>/api`.
//
// Route: POST /api/chat/completions   (Vercel maps this file to that path)
// ---------------------------------------------------------------------------
export const config = { runtime: 'edge' };

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

// Manifest translates OpenAI's loose `{ type: 'json_object' }` shorthand to
// Anthropic native structured output. Anthropic requires every object in that
// schema to declare `additionalProperties: false`; an unconstrained
// `{ type: 'object' }` is rejected before the model runs. RoleReveal has one
// JSON-mode operation, so upgrade that shorthand to its real strict schema at
// the proxy. This fixes existing extension installs immediately and remains
// compatible when Manifest routes the same request to Gemini or OpenAI.
const EVALUATION_RESPONSE_FORMAT = {
  type: 'json_schema',
  json_schema: {
    name: 'rolereveal_evaluation',
    strict: true,
    schema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        perResume: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              label: { type: 'string' },
              score: { type: 'number' },
            },
            required: ['label', 'score'],
          },
        },
        bestResume: { type: 'string' },
        overallScore: { type: 'number' },
        verdict: { type: 'string', enum: ['Apply', 'Maybe', 'Skip'] },
        summary: { type: 'string' },
        dimensions: {
          type: 'object',
          additionalProperties: false,
          properties: {
            skills: { type: 'number' },
            experience: { type: 'number' },
            roleContext: { type: 'number' },
          },
          required: ['skills', 'experience', 'roleContext'],
        },
        whyMatch: { type: 'string' },
        watchOuts: { type: 'string' },
      },
      required: [
        'perResume',
        'bestResume',
        'overallScore',
        'verdict',
        'summary',
        'dimensions',
        'whyMatch',
        'watchOuts',
      ],
    },
  },
} as const;

export function normalizeResponseFormat(value: unknown): unknown {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return value;
  const type = (value as Record<string, unknown>).type;
  return type === 'json_object' ? EVALUATION_RESPONSE_FORMAT : value;
}

// ── Strict, persistent rate limiting via Upstash Redis ──────────────────────
// Counters survive across edge invocations. Enforced ONLY when Upstash env vars
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

export default async function handler(req: Request): Promise<Response> {
  if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: CORS });
  if (req.method !== 'POST') return json({ error: 'method_not_allowed' }, 405);

  const UPSTREAM = process.env.UPSTREAM_BASE_URL; // e.g. https://generativelanguage.googleapis.com/v1beta/openai
  const KEY = process.env.UPSTREAM_API_KEY; // the REAL provider/gateway key — server-only
  if (!UPSTREAM || !KEY) return json({ error: 'server_not_configured' }, 500);

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

  // Optional model allowlist — keeps all traffic on cheap models.
  const allowed = (process.env.ALLOWED_MODELS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const model = body.model || process.env.DEFAULT_MODEL || 'auto';
  if (allowed.length && !allowed.includes(model)) {
    return json({ error: 'model_not_allowed', model }, 400);
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

  const baseBody: Record<string, unknown> = {
    model,
    messages,
    temperature: clampNum(body.temperature, 0, 2, 0.2),
    // Floor of 1536: Anthropic's extended-thinking minimum budget is 1024
    // tokens, and thinking tokens count against max_tokens, so anything at or
    // below that floor 400s outright ("max_tokens must be greater than
    // thinking.budget_tokens") whenever Manifest's fallback chain lands on a
    // thinking-enabled Claude model (e.g. gemini-2.5-flash 429s -> claude-haiku-4-5).
    max_tokens: Math.min(Math.max(clampNum(body.max_tokens, 1, 4096, 900), 1536), 3072),
  };
  const responseFormat = normalizeResponseFormat(body.response_format);
  const upstreamBody: Record<string, unknown> = {
    ...baseBody,
    ...(responseFormat ? { response_format: responseFormat } : {}),
  };

  const call = (b: Record<string, unknown>) =>
    fetch(`${UPSTREAM.replace(/\/+$/, '')}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${KEY}` },
      body: JSON.stringify(b),
    });

  // The body we actually sent, so each retry below keeps earlier adjustments.
  let sent: Record<string, unknown> = upstreamBody;
  let upstream: Response;
  let text: string;
  try {
    upstream = await call(sent);
    text = await upstream.text();
  } catch (e: any) {
    return json({ error: 'upstream_unreachable', detail: String(e?.message || e) }, 502);
  }

  // Upstream failures are otherwise invisible: the body is passed straight back
  // to the extension, which shows only the human-readable message. Log the shape
  // so the gateway's own error payload is diagnosable from `vercel logs`.
  const logUpstreamFailure = (label: string) => {
    if (upstream.status < 400) return;
    console.error(
      `[upstream ${label}] status=${upstream.status} max_tokens=${sent.max_tokens} body=${text.slice(0, 400)}`,
    );
  };
  logUpstreamFailure('attempt-1');

  // Safety net: Manifest injects Anthropic thinking on the OpenAI-compatible
  // /chat/completions path (its "strip adaptive thinking for Claude Haiku" fix
  // only covers native Anthropic Messages requests), and thinking tokens count
  // against max_tokens. When the route's configured budget is larger than the
  // 1536 floor above, Anthropic rejects the call pre-model with 0 tokens used,
  // the whole chain is exhausted, and Manifest reports the PRIMARY's failure —
  // typically the 429 that triggered the fallback in the first place, with no
  // mention of budget_tokens. So retry on that 429 too, not just on a body that
  // names the thinking error. The real fix is unsetting thinking on that route's
  // Model params in Manifest; this only keeps a rate-limited primary survivable.
  const thinkingRetryMaxTokens = clampNum(
    process.env.THINKING_RETRY_MAX_TOKENS,
    2048,
    32_000,
    16_384,
  );
  // Safety net: a reasoning-capable fallback (e.g. deepseek-v4-flash, nvidia
  // nemotron — landed on after the primary 429s, or picked directly) can spend
  // its whole max_tokens budget on hidden reasoning tokens and get cut off by
  // finish_reason "length" before ever finishing the visible answer. Manifest
  // reports that as a clean 200, so none of the checks above fire, and it shows
  // up two ways downstream: choices[0].message.content === "" ("the provider
  // returned an empty response"), or a half-written answer with no closing JSON
  // brace ("No JSON object found in LLM response"). Both are the same
  // truncation, so key off finish_reason directly rather than guessing from
  // content shape, and give it the same headroom retry as the thinking-budget
  // case above.
  const wasTruncatedByLength = (): boolean => {
    if (upstream.status !== 200) return false;
    try {
      const choice = JSON.parse(text)?.choices?.[0];
      return choice?.finish_reason === 'length' || choice?.native_finish_reason === 'length';
    } catch {
      return false;
    }
  };
  if (wasTruncatedByLength()) {
    console.error(
      `[upstream truncated] max_tokens=${sent.max_tokens} body=${text.slice(0, 400)}`,
    );
  }

  const worthRetryingWithRoomForThinking =
    upstream.status === 429 ||
    (upstream.status >= 400 && text.includes('thinking.budget_tokens')) ||
    wasTruncatedByLength();
  if (worthRetryingWithRoomForThinking && Number(sent.max_tokens) < thinkingRetryMaxTokens) {
    sent = { ...sent, max_tokens: thinkingRetryMaxTokens };
    try {
      upstream = await call(sent);
      text = await upstream.text();
    } catch (e: any) {
      return json({ error: 'upstream_unreachable', detail: String(e?.message || e) }, 502);
    }
    logUpstreamFailure('thinking-retry');
  }

  // Safety net: if Manifest still exhausts its Anthropic route because a
  // provider rejects structured output, retry once without response_format.
  // The system prompt already requires JSON-only output and extractJson()
  // tolerates the resulting plain-text-mode response.
  const looksLikeManifestAnthropicSchemaBug = (): boolean => {
    if (upstream.status !== 400 || !body.response_format) return false;
    try {
      const err = JSON.parse(text)?.error;
      if (err?.code !== 'fallback_exhausted' || err?.source !== 'manifest') return false;
      const fallbacks = Array.isArray(err?.attempted_fallbacks) ? err.attempted_fallbacks : [];
      return err?.provider === 'anthropic' || fallbacks.some((f: any) => f?.provider === 'anthropic');
    } catch {
      return false;
    }
  };

  if (looksLikeManifestAnthropicSchemaBug()) {
    const { response_format: _dropped, ...withoutSchema } = sent;
    sent = withoutSchema;
    try {
      upstream = await call(sent);
      text = await upstream.text();
    } catch (e: any) {
      return json({ error: 'upstream_unreachable', detail: String(e?.message || e) }, 502);
    }
    logUpstreamFailure('schema-retry');
  }

  // Pass the OpenAI-shaped response straight through (extension reads choices[0]).
  return new Response(text, {
    status: upstream.status,
    headers: { 'content-type': 'application/json', ...CORS },
  });
}

// whiz-store — turns a Paddle payment into a Whiz license key.
//
//   POST /paddle/webhook          Paddle's transaction.completed
//   GET  /license?transaction_id  the thank-you page asks for its key
//
// Both go through one idempotent `licenseFor`: the key is issued once per
// transaction, stored in KV under the transaction id, and read back on
// every later call. Paddle delivers webhooks at least once and the page
// may ask before the webhook lands, so whichever comes first issues and
// the other reads; a retry never mints a second key.
//
// Bindings (wrangler.toml): LICENSES (KV). Secrets (`wrangler secret put`,
// never in the repo): PADDLE_API_KEY, PADDLE_WEBHOOK_SECRET,
// LICENSE_SIGNING_KEY (the base64 seed from whiz-license keygen).
// Vars: PADDLE_ENV ("sandbox" | "production"), ALLOWED_ORIGIN.

import { importSigningKey, issue } from "./license.js";
import { getTransaction, verifySignature } from "./paddle.js";

// A transaction that Paddle has taken payment for. One-time purchases go
// paid → completed within seconds; both are paid for.
const PAID = new Set(["paid", "completed"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (request.method === "POST" && url.pathname === "/paddle/webhook") return await webhook(request, env);
      if (url.pathname === "/license") {
        if (request.method === "OPTIONS") return cors(env, new Response(null, { status: 204 }));
        if (request.method === "GET") return cors(env, await lookup(url, env));
      }
      return json({ error: "not found" }, 404);
    } catch (error) {
      console.error("whiz-store:", error);
      // Any non-2xx makes Paddle retry, which is what a transient failure
      // (Paddle API, KV) wants.
      return json({ error: "internal error" }, 500);
    }
  },
};

async function webhook(request, env) {
  const raw = await request.text();
  if (!(await verifySignature(request.headers.get("Paddle-Signature"), raw, env.PADDLE_WEBHOOK_SECRET))) {
    // Not 2xx, so a rotated secret still gets retried once redeployed.
    return json({ error: "bad signature" }, 401);
  }
  const event = JSON.parse(raw);
  if (event.event_type !== "transaction.completed") return json({ ignored: event.event_type });
  const result = await licenseFor(env, event.data?.id);
  // Not ours (another product) or not paid: acknowledged, nothing to issue.
  return json({ transaction_id: event.data?.id, issued: result.status === "issued" });
}

async function lookup(url, env) {
  const transactionId = url.searchParams.get("transaction_id") ?? "";
  if (!/^txn_[a-z0-9]{26}$/.test(transactionId)) return json({ error: "transaction_id required" }, 400);
  const result = await licenseFor(env, transactionId);
  switch (result.status) {
    // The key and the name it shows; nothing else about the buyer leaves KV.
    case "issued": return json({ key: result.record.key, name: result.record.name });
    // The page retries for a few seconds: payment can still be settling.
    case "pending": return json({ status: "pending" }, 202);
    default: return json({ error: "no license for this transaction" }, 404);
  }
}

/**
 * The license for a transaction: read from KV, or issued now when Paddle
 * says it is paid for a Whiz price. Returns { status: "issued", record },
 * { status: "pending" } or { status: "none" }.
 */
export async function licenseFor(env, transactionId, fetchImpl = fetch, today = new Date()) {
  if (!transactionId) return { status: "none" };
  const stored = await env.LICENSES.get(transactionId, "json");
  if (stored) return { status: "issued", record: stored };

  const transaction = await getTransaction(env, transactionId, fetchImpl);
  if (!transaction) return { status: "none" };
  const major = licenseMajor(transaction);
  if (!major) return { status: "none" };
  if (!PAID.has(transaction.status)) return { status: "pending" };

  const customer = transaction.customer ?? {};
  const email = customer.email ?? "";
  // The checkout asks for an email, a name only sometimes; the app shows
  // "Licensed to <name>", and an email reads fine there.
  const name = (customer.name ?? "").trim() || email || "Whiz customer";
  const issued = (transaction.billed_at ?? transaction.created_at ?? today.toISOString()).slice(0, 10);
  const product = productFor(env);
  const key = await issue({ id: transactionId, name, issued, major, product }, await importSigningKey(env.LICENSE_SIGNING_KEY));
  const record = { key, name, email, issued, major, product, customer_id: transaction.customer_id ?? null };

  // A concurrent first call (webhook and page at once) may also issue: the
  // two keys differ only if the inputs did, and either verifies. Last write
  // wins and later reads agree.
  await env.LICENSES.put(transactionId, JSON.stringify(record));
  return { status: "issued", record };
}

/**
 * Sandbox purchases are paid with test cards, so their keys must not open
 * Whiz: they are signed for "whiz-sandbox", which the app verifies and then
 * refuses as another product. That still proves the whole chain (signature
 * included) in a test purchase, and a found `?store=sandbox` buys nothing.
 */
export function productFor(env) {
  return env.PADDLE_ENV === "production" ? "whiz" : "whiz-sandbox";
}

/** The license major version a transaction buys, from its Whiz price; 0 if none. */
export function licenseMajor(transaction) {
  for (const item of transaction.items ?? []) {
    const major = Number(item.price?.custom_data?.license_major);
    if (Number.isInteger(major) && major >= 1) return major;
  }
  return 0;
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function cors(env, response) {
  response.headers.set("Access-Control-Allow-Origin", env.ALLOWED_ORIGIN ?? "https://duduphudu.app");
  response.headers.set("Access-Control-Allow-Methods", "GET, OPTIONS");
  response.headers.set("Vary", "Origin");
  return response;
}

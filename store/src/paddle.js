// The two things the store needs from Paddle: trusting a webhook, and
// reading a transaction.

// Paddle re-signs every retry with a fresh timestamp, so a generous window
// costs nothing; it only bounds how long a captured request stays usable,
// and replaying one here re-issues the same key (idempotent) anyway.
export const SIGNATURE_TOLERANCE_SECONDS = 300;

/**
 * Verify a `Paddle-Signature` header (`ts=…;h1=…[;h1=…]`) against the raw
 * body: HMAC-SHA256 of `${ts}:${rawBody}` keyed with the destination's
 * secret, hex-encoded. More than one h1 appears while a secret rotates; any
 * match passes. Returns true or false, never throws on bad input.
 */
export async function verifySignature(header, rawBody, secret, nowSeconds = Math.floor(Date.now() / 1000)) {
  if (!header || !secret) return false;
  let ts = null;
  const signatures = [];
  for (const part of header.split(";")) {
    const at = part.indexOf("=");
    if (at < 0) continue;
    const key = part.slice(0, at).trim();
    const value = part.slice(at + 1).trim();
    if (key === "ts") ts = value;
    else if (key === "h1") signatures.push(value.toLowerCase());
  }
  if (!ts || !/^\d+$/.test(ts) || signatures.length === 0) return false;
  if (Math.abs(nowSeconds - Number(ts)) > SIGNATURE_TOLERANCE_SECONDS) return false;

  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`${ts}:${rawBody}`)));
  const expected = [...mac].map((b) => b.toString(16).padStart(2, "0")).join("");
  return signatures.some((candidate) => timingSafeEqual(candidate, expected));
}

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export function apiBase(env) {
  return env.PADDLE_ENV === "production" ? "https://api.paddle.com" : "https://sandbox-api.paddle.com";
}

/** A transaction with its customer, or null when Paddle has no such id. */
export async function getTransaction(env, transactionId, fetchImpl = fetch) {
  const response = await fetchImpl(
    `${apiBase(env)}/transactions/${encodeURIComponent(transactionId)}?include=customer`,
    { headers: { Authorization: `Bearer ${env.PADDLE_API_KEY}`, "Paddle-Version": "1" } });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Paddle API ${response.status} for ${transactionId}`);
  return (await response.json()).data;
}

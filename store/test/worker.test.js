import { test } from "node:test";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import worker, { licenseFor, licenseMajor } from "../src/worker.js";
import { VECTOR } from "./vector.js";

const TXN = "txn_01h0000000000000000000000a";

function kv() {
  const map = new Map();
  return { map, async get(k, type) { const v = map.get(k); return v == null ? null : type === "json" ? JSON.parse(v) : v; },
           async put(k, v) { map.set(k, v); } };
}

function environment(overrides = {}) {
  return { LICENSES: kv(), PADDLE_ENV: "sandbox", PADDLE_API_KEY: "pdl_sdbx_apikey_test",
           PADDLE_WEBHOOK_SECRET: "pdl_ntfset_test", LICENSE_SIGNING_KEY: VECTOR.seed,
           ALLOWED_ORIGIN: "https://duduphudu.app", ...overrides };
}

function transaction(fields = {}) {
  return { id: TXN, status: "completed", customer_id: "ctm_1", billed_at: "2026-10-21T10:00:00Z",
           customer: { email: "ada@example.com", name: "Ada Lovelace" },
           items: [{ price: { id: "pri_1", custom_data: { license_major: 1 } } }], ...fields };
}

/** A fake Paddle API answering GET /transactions/:id; counts calls. */
function paddle(txn) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, auth: init.headers.Authorization });
    if (!txn) return new Response("{}", { status: 404 });
    return new Response(JSON.stringify({ data: txn }), { status: 200 });
  };
  return { calls, fetchImpl };
}

test("a paid Whiz transaction gets one key, stored and reused", async () => {
  const env = environment();
  const api = paddle(transaction());
  const first = await licenseFor(env, TXN, api.fetchImpl);
  assert.equal(first.status, "issued");
  assert.match(first.record.key, /^WHIZ1-/);
  assert.equal(first.record.name, "Ada Lovelace");
  assert.equal(first.record.issued, "2026-10-21");
  assert.equal(api.calls[0].url, `https://sandbox-api.paddle.com/transactions/${TXN}?include=customer`);
  assert.equal(api.calls[0].auth, "Bearer pdl_sdbx_apikey_test");

  const again = await licenseFor(env, TXN, api.fetchImpl);
  assert.equal(again.record.key, first.record.key);
  assert.equal(api.calls.length, 1, "the second call reads KV, not Paddle");
});

test("the key is deterministic: a concurrent second issue would be the same bytes", async () => {
  const a = await licenseFor(environment(), TXN, paddle(transaction()).fetchImpl);
  const b = await licenseFor(environment(), TXN, paddle(transaction()).fetchImpl);
  assert.equal(a.record.key, b.record.key);
});

test("no name at checkout: the email is the licensee", async () => {
  const result = await licenseFor(environment(), TXN, paddle(transaction({ customer: { email: "ada@example.com", name: null } })).fetchImpl);
  assert.equal(result.record.name, "ada@example.com");
});

test("unpaid is pending; another product or an unknown id gets nothing", async () => {
  assert.equal((await licenseFor(environment(), TXN, paddle(transaction({ status: "billed" })).fetchImpl)).status, "pending");
  assert.equal((await licenseFor(environment(), TXN, paddle(transaction({ items: [{ price: { custom_data: null } }] })).fetchImpl)).status, "none");
  assert.equal((await licenseFor(environment(), TXN, paddle(null).fetchImpl)).status, "none");
});

test("license major comes from the price", () => {
  assert.equal(licenseMajor(transaction()), 1);
  assert.equal(licenseMajor(transaction({ items: [{ price: { custom_data: { license_major: "2" } } }] })), 2);
  assert.equal(licenseMajor(transaction({ items: [] })), 0);
});

// --- HTTP surface, with fetch stubbed globally for the Paddle API ---

function stubFetch(txn) {
  const original = globalThis.fetch;
  globalThis.fetch = paddle(txn).fetchImpl;
  return () => { globalThis.fetch = original; };
}

function signed(body, secret = "pdl_ntfset_test") {
  const ts = Math.floor(Date.now() / 1000);
  const h1 = createHmac("sha256", secret).update(`${ts}:${body}`).digest("hex");
  return new Request("https://store.duduphudu.app/paddle/webhook",
    { method: "POST", body, headers: { "Paddle-Signature": `ts=${ts};h1=${h1}` } });
}

test("webhook: a signed transaction.completed issues the key", async () => {
  const restore = stubFetch(transaction());
  try {
    const env = environment();
    const response = await worker.fetch(signed(JSON.stringify({ event_type: "transaction.completed", data: { id: TXN } })), env);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { transaction_id: TXN, issued: true });
    assert.ok(env.LICENSES.map.has(TXN));

    // The thank-you page then reads the same key.
    const page = await worker.fetch(new Request(`https://store.duduphudu.app/license?transaction_id=${TXN}`), env);
    assert.equal(page.status, 200);
    assert.equal(page.headers.get("Access-Control-Allow-Origin"), "https://duduphudu.app");
    const body = await page.json();
    assert.equal(body.key, JSON.parse(env.LICENSES.map.get(TXN)).key);
    assert.deepEqual(Object.keys(body).sort(), ["key", "name"], "no email in the response");
  } finally { restore(); }
});

test("webhook: a bad signature is refused (non-2xx, so Paddle retries) and issues nothing", async () => {
  const env = environment();
  const response = await worker.fetch(signed(JSON.stringify({ event_type: "transaction.completed", data: { id: TXN } }), "wrong"), env);
  assert.equal(response.status, 401);
  assert.equal(env.LICENSES.map.size, 0);
});

test("webhook: other events are acknowledged and ignored", async () => {
  const response = await worker.fetch(signed(JSON.stringify({ event_type: "customer.created", data: { id: "ctm_1" } })), environment());
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ignored: "customer.created" });
});

test("license lookup: pending while payment settles, 400 on a malformed id, 500 when Paddle fails", async () => {
  let restore = stubFetch(transaction({ status: "billed" }));
  try {
    const pending = await worker.fetch(new Request(`https://store.duduphudu.app/license?transaction_id=${TXN}`), environment());
    assert.equal(pending.status, 202);
  } finally { restore(); }
  const bad = await worker.fetch(new Request("https://store.duduphudu.app/license?transaction_id=../../etc"), environment());
  assert.equal(bad.status, 400);

  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response("down", { status: 503 });
  try {
    const failed = await worker.fetch(new Request(`https://store.duduphudu.app/license?transaction_id=${TXN}`), environment());
    assert.equal(failed.status, 500);
  } finally { globalThis.fetch = original; }
});

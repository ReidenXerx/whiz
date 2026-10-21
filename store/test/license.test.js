import { test } from "node:test";
import assert from "node:assert/strict";
import { base64Decode, importSigningKey, issue, payloadBytes } from "../src/license.js";
import { VECTOR } from "./vector.js";

test("the shared vector: same bytes as the Swift issuer", async () => {
  const key = await issue(VECTOR.license, await importSigningKey(VECTOR.seed));
  assert.equal(key, VECTOR.key);
});

test("an issued key verifies against the public key", async () => {
  const [payload, signature] = VECTOR.key.slice("WHIZ1-".length).split(".");
  const unb64url = (s) => base64Decode(s.replaceAll("-", "+").replaceAll("_", "/") + "=".repeat((4 - (s.length % 4)) % 4));
  const publicKey = await crypto.subtle.importKey("raw", base64Decode(VECTOR.publicKey), { name: "Ed25519" }, false, ["verify"]);
  assert.ok(await crypto.subtle.verify({ name: "Ed25519" }, publicKey, unb64url(signature), unb64url(payload)));
  assert.deepEqual(JSON.parse(new TextDecoder().decode(unb64url(payload))), { ...VECTOR.license, product: "whiz" });
});

test("the payload is sorted, compact, slashes and non-ASCII unescaped", () => {
  const text = new TextDecoder().decode(payloadBytes({ name: "A/B Ø", major: 1, issued: "2026-01-02", id: "x" }));
  assert.equal(text, '{"id":"x","issued":"2026-01-02","major":1,"name":"A/B Ø","product":"whiz"}');
});

test("incomplete licenses and wrong seeds are refused", async () => {
  assert.throws(() => payloadBytes({ id: "x", name: "", issued: "2026-01-02", major: 1 }));
  assert.throws(() => payloadBytes({ id: "x", name: "n", issued: "21.10.2026", major: 1 }));
  assert.throws(() => payloadBytes({ id: "x", name: "n", issued: "2026-01-02", major: 0 }));
  await assert.rejects(importSigningKey("AAAA"));
});

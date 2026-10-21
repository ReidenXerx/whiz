import { test } from "node:test";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { verifySignature } from "../src/paddle.js";

const secret = "pdl_ntfset_test_secret";
const body = '{"event_type":"transaction.completed","data":{"id":"txn_x"}}';
const sign = (ts, raw = body, key = secret) => createHmac("sha256", key).update(`${ts}:${raw}`).digest("hex");

test("a correctly signed body passes", async () => {
  const ts = 1_800_000_000;
  assert.ok(await verifySignature(`ts=${ts};h1=${sign(ts)}`, body, secret, ts + 10));
});

test("any of several h1 values passes (secret rotation)", async () => {
  const ts = 1_800_000_000;
  assert.ok(await verifySignature(`ts=${ts};h1=${"0".repeat(64)};h1=${sign(ts)}`, body, secret, ts));
});

test("a changed body, wrong secret, stale timestamp or junk header fails", async () => {
  const ts = 1_800_000_000;
  const header = `ts=${ts};h1=${sign(ts)}`;
  assert.equal(await verifySignature(header, body + " ", secret, ts), false);
  assert.equal(await verifySignature(header, body, "another", ts), false);
  assert.equal(await verifySignature(header, body, secret, ts + 301), false);
  for (const junk of ["", "nonsense", `ts=${ts}`, `h1=${sign(ts)}`, `ts=abc;h1=${sign(ts)}`]) {
    assert.equal(await verifySignature(junk, body, secret, ts), false, junk);
  }
  assert.equal(await verifySignature(header, body, "", ts), false);
});

// Whiz license keys, signed the way the app verifies them.
//
// The format and its rules live in macos/Sources/WhizLicense/LicenseKey.swift;
// this is the same signer, so the store can issue keys without the seller's
// Mac:
//
//     WHIZ1-<base64url(payload JSON)>.<base64url(Ed25519 signature)>
//
// The app checks the signature over the payload bytes exactly as encoded,
// so field order does not affect validity. The payload is kept identical to
// the Swift encoder's anyway (sorted keys, no spaces, slashes unescaped).
// Signatures are not compared: WebCrypto's Ed25519 is deterministic, while
// CryptoKit adds randomness, and both verify. test/vector.js and the Swift
// LicenseKeyTests pin this with one shared test vector.

export const PREFIX = "WHIZ1-";
export const PRODUCT = "whiz";

// An Ed25519 private key as PKCS#8 is this fixed DER header followed by the
// 32-byte seed — the raw form `whiz-license keygen` stores (base64).
const PKCS8_ED25519_PREFIX = Uint8Array.from([
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20,
]);

/** Import the base64 seed from ~/.config/whiz-license/signing.key. */
export async function importSigningKey(seedBase64) {
  const seed = base64Decode(seedBase64.trim());
  if (seed.length !== 32) throw new Error("the signing key must be a 32-byte Ed25519 seed (base64)");
  const pkcs8 = new Uint8Array(PKCS8_ED25519_PREFIX.length + seed.length);
  pkcs8.set(PKCS8_ED25519_PREFIX);
  pkcs8.set(seed, PKCS8_ED25519_PREFIX.length);
  return crypto.subtle.importKey("pkcs8", pkcs8, { name: "Ed25519" }, false, ["sign"]);
}

/** The payload bytes: the Swift encoder's output for the same license. */
export function payloadBytes({ id, name, issued, major, product = PRODUCT }) {
  if (!id || !name || !/^\d{4}-\d{2}-\d{2}$/.test(issued) || !Number.isInteger(major) || major < 1) {
    throw new Error("a license needs id, name, issued (yyyy-MM-dd) and a major version of 1 or more");
  }
  // Insertion order = sorted key order; JSON.stringify keeps it.
  const ordered = { id, issued, major, name, product };
  return new TextEncoder().encode(JSON.stringify(ordered));
}

/** Sign a license into its text form. */
export async function issue(license, signingKey) {
  const payload = payloadBytes(license);
  const signature = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, signingKey, payload));
  return PREFIX + base64url(payload) + "." + base64url(signature);
}

export function base64url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

export function base64Decode(text) {
  const binary = atob(text);
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

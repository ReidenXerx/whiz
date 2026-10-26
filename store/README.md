# whiz-store

A Cloudflare Worker that turns a Paddle payment into a Whiz license key.

- Paddle sends `transaction.completed` to `POST /paddle/webhook`.
- The thank-you page asks `GET /license?transaction_id=txn_…`.

Whichever comes first issues the key. It is signed exactly as `whiz-license issue` would (see `src/license.js`), stored in KV under the transaction id, and returned on every later call. Retries never mint a second key.

## Test

```bash
npm test        # node --test; no dependencies
```

`test/vector.js` is shared with `macos/Tests/WhizLicenseTests`: the app must accept a key this signer produces.

## Deploy

All of this runs on the Cloudflare account that holds `duduphudu.app`. The same code is deployed twice, once per Paddle environment, each with its own KV, secrets and webhook destination, so a test purchase never touches a real one:

| | Live | Sandbox |
|---|---|---|
| Worker | `whiz-store` | `whiz-store-sandbox` |
| Host | `store.duduphudu.app` | `store-sandbox.duduphudu.app` |
| Deploy | `npx wrangler deploy` | `npx wrangler deploy --env sandbox` |
| Keys it issues | `whiz` | `whiz-sandbox` (the app refuses them) |

Each deployment needs three secrets (`npx wrangler secret put NAME`, plus `--env sandbox` for the sandbox one; pipe them in, never paste them anywhere):

- `PADDLE_API_KEY`: that environment's API key (`pdl_live_apikey_…` / `pdl_sdbx_apikey_…`), with read access to transactions and customers.
- `PADDLE_WEBHOOK_SECRET`: the secret of that environment's notification destination, `https://<host>/paddle/webhook` for `transaction.completed`.
- `LICENSE_SIGNING_KEY`: the contents of `~/.config/whiz-license/signing.key`, the same for both.

The site (`docs/index.html`, `STORE`) points each checkout at its own host.

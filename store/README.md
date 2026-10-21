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

All of this runs on the Cloudflare account that holds `duduphudu.app`.

```bash
npx wrangler login
npx wrangler kv namespace create LICENSES        # put the id into wrangler.toml
npx wrangler secret put PADDLE_API_KEY           # pdl_sdbx_apikey_… (sandbox) / pdl_live_apikey_…
npx wrangler secret put PADDLE_WEBHOOK_SECRET    # the notification destination's secret
npx wrangler secret put LICENSE_SIGNING_KEY      # contents of ~/.config/whiz-license/signing.key
npx wrangler deploy                              # → https://store.duduphudu.app
```

Then, in Paddle, add a notification destination at `https://store.duduphudu.app/paddle/webhook` for `transaction.completed`. Its secret is `PADDLE_WEBHOOK_SECRET`.

Sandbox and live use separate keys, secrets and destinations. Going live means setting `PADDLE_ENV = "production"` and putting in the live secrets.

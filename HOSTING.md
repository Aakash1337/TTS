# Hosting the TTS Reader beyond this PC

The app runs on this machine (it needs the GPU for Chatterbox/OCR and Ollama for
the AI features); Cloudflare provides the public doorway. Two levels:

## Level 1 — instant, ephemeral (works today)

Double-click **`Start TTS Reader (Online).bat`**. A random
`https://<words>.trycloudflare.com` URL appears in the tunnel window — open it
from any device.

- ✅ Free, zero configuration, no account needed.
- ⚠️ URL changes every launch, and there is **no login** — anyone holding the
  URL can use the app while it's up. Fine for personal, occasional use; don't
  share the URL.

## Level 2 — permanent URL + login (recommended for regular use)

Needs a (free) Cloudflare account and a domain added to it. One-time setup,
~10 minutes, run in PowerShell:

```powershell
# 1. Authenticate cloudflared with your Cloudflare account (opens the browser)
cloudflared tunnel login

# 2. Create a named tunnel
cloudflared tunnel create tts-reader

# 3. Route a hostname on your domain to it (pick any subdomain you like)
cloudflared tunnel route dns tts-reader tts.YOURDOMAIN.com
```

Then create `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: tts-reader
credentials-file: C:\Users\blis\.cloudflared\<TUNNEL-ID>.json
ingress:
  - hostname: tts.YOURDOMAIN.com
    service: http://127.0.0.1:8756
  - service: http_status:404
```

Run it with `cloudflared tunnel run tts-reader` (or install as a Windows
service so it starts with the PC: `cloudflared service install`).

**Add the login gate (important):** in the Cloudflare dashboard →
*Zero Trust → Access → Applications → Add application → Self-hosted*, protect
`tts.YOURDOMAIN.com`, policy = "Emails ending in / equal to your email", login
method = One-time PIN. After this, opening the app anywhere prompts for a code
sent to your email — nobody else gets in. Free for personal use.

## Why not host on a cloud box?

Aria/edge voices would work on a $5 VPS, but Chatterbox (GPU), Gemma
(auto-mode/summary/polish), and GPU OCR would not. The tunnel keeps 100% of the
features by serving from this machine. Trade-off: the PC must be on.

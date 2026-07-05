@echo off
rem Start the TTS Reader AND share it over the internet via a Cloudflare quick
rem tunnel. The public https://....trycloudflare.com URL appears in the second
rem window - open it on your phone or any device.
rem
rem NOTES
rem  - The URL is RANDOM and changes every launch (quick tunnels are ephemeral).
rem  - There is NO login on a quick tunnel: anyone who has the URL can use the
rem    app while it runs. Don't post the URL publicly; close the windows to stop.
rem  - For a PERMANENT url + Cloudflare Access login, see HOSTING.md.
start "TTS Reader server" "E:\TTS\.venv\Scripts\python.exe" "E:\TTS\serve.py" --no-browser --port 8756
timeout /t 3 /nobreak >nul
start "Cloudflare tunnel (public URL below)" cloudflared tunnel --url http://127.0.0.1:8756

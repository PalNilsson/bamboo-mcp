# Remote Testing: Bamboo MCP from Home via SSH Tunnel

This guide explains how to run the Bamboo MCP server and Streamlit UI on lxplus
and access them from your home machine over a CERN VPN connection.

## Prerequisites

- CERN VPN active on your home machine
- SSH key installed on lxplus (see below if not done yet)

---

## One-time setup: Install your SSH key on lxplus

This avoids re-entering your password for every tunnel session.

```bash
# On your home machine:
ssh-keygen -t ed25519
ssh-copy-id user_name@lxplus.cern.ch
```

> **Note:** CERN lxplus still prompts for your password even with a key installed,
> due to 2FA enforcement for connections from outside CERN. This is expected.

---

## Step 1 — Open the SSH tunnel (home machine)

In a dedicated terminal on your home machine, run:

```bash
ssh -L 8000:localhost:8000 -L 8501:localhost:8501 user_name@lxplus947.cern.ch -N
```

- `-L 8000:localhost:8000` — forwards MCP server port
- `-L 8501:localhost:8501` — forwards Streamlit UI port
- `-N` — no shell, just port forwarding

Enter your CERN password when prompted. The terminal will appear to hang with no
output — that is correct. **Leave it open for the duration of your session.**

---

## Step 2 — Start the MCP server (lxplus)

In a separate SSH session to lxplus:

```bash
cd ~/bamboo/bamboo-mcp
source ../venv/bin/activate
source ../bamboo_config/bamboo_env.sh

python -m bamboo.server_http --host 127.0.0.1 --port 8000
```

Expected output:

```
Bamboo MCP HTTP server  v1.1.0
  MCP endpoint : http://127.0.0.1:8000/mcp
  Health check : http://127.0.0.1:8000/healthz
  Workers      : 1
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Step 3 — Start the Streamlit UI (lxplus)

In another SSH session to lxplus:

```bash
cd ~/bamboo/bamboo-mcp
source ../venv/bin/activate
source ../bamboo_config/bamboo_env.sh

streamlit run interfaces/streamlit/chat.py --server.port 8501 --server.address 127.0.0.1
```

---

## Step 4 — Verify and connect (home machine)

Check the MCP server is reachable:

```bash
curl http://localhost:8000/healthz
# Expected: ok
```

Then open the Streamlit UI in your browser:

```
http://localhost:8501
```

Point the MCP URL inside the app to:

```
http://localhost:8000/mcp
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `curl: (7) Failed to connect to lxplus…:8000` | Firewall blocks direct access | Always use `localhost` via tunnel, never the lxplus hostname directly |
| `NS_ERROR_NET_EMPTY_RESPONSE` on port 8501 | Streamlit not started yet | Run Step 3 |
| `Not Acceptable: Client must accept text/event-stream` on `/mcp` | Expected — browser hit the MCP endpoint directly | Use `curl localhost:8000/healthz` to health-check instead |
| SSH tunnel prompts for password twice | 2FA enforcement on lxplus from outside CERN | Enter CERN password at first prompt; leave second prompt blank or enter OTP if enrolled |
| Tunnel terminal hangs silently after password | Correct behaviour with `-N` flag | Leave it open; silence means it is working |

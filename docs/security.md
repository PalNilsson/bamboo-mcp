# Authentication & Security

Bamboo supports optional Bearer token authentication for HTTP-based
entrypoints. This is intended for internal deployments (e.g. CERN machines)
where controlled access to the MCP endpoint is required.

Authentication is enforced only for HTTP transports. The stdio entrypoint
does not use HTTP headers and therefore does not enforce Bearer tokens.

---

## Overview

When enabled, clients must send:

```Authorization: Bearer <token>```

on each HTTP request to `/mcp`.

If authentication is configured:

- Missing header → `401 Unauthorized`
- Malformed header → `401 Unauthorized`
- Invalid token → `403 Forbidden`

If no tokens are configured, authentication is disabled (useful for local development).

---

## Enabling Authentication

Authentication is enabled automatically when one of the following environment
variables is set:

### Option 1: Tokens file (recommended)

```BAMBOO_MCP_TOKENS_FILE=/path/to/tokens.txt```

Example `tokens.txt`:

Format: client_id: token

```
bamboo-cli: AAA...
bamboo-ui: BBB...
bamboo-ci: CCC...
```
Recommended permissions:

```chmod 600 tokens.txt```

### Option 2: Inline tokens

```BAMBOO_MCP_TOKENS="bamboo-cli:AAA,bamboo-ui:BBB"```


This is convenient for CI or containerized environments.

---

## Generating Tokens

Use the provided script:

```aiignore
python scripts/generate_tokens.py
    --client bamboo-cli
    --client bamboo-ui
    --bits 256
    --format tokens.txt
```


Default entropy is **256 bits**, which is cryptographically strong.

Tokens are generated using Python's `secrets` module.

---

## Security Model

### Token Properties

- 256-bit entropy by default
- Generated using `secrets.token_urlsafe`
- Compared using constant-time comparison (`secrets.compare_digest`)
- Never logged by the server

### Design Characteristics

- Stateless (each request must include the token)
- Tokens are loaded once at server startup
- Multiple tokens supported (per-client tokens recommended)
- Authentication is automatically disabled if no tokens are configured

---

## REST analysis API

The `/api/v1` REST surface enforces the same policy as `/mcp`: both call
`bamboo.entrypoints.rest.authenticate()`, so there is one allowlist and one
implementation rather than two things to forget to update. A token that works
for MCP works for REST.

Two consequences worth stating plainly:

- **Give the PanDA monitor its own client id** (`panda-monitor: <token>`) so
  service traffic is distinguishable from a human's in the logs, and so the
  monitor's token can be rotated without touching anyone else's.
- **When no tokens are configured, authentication is disabled and every caller
  is recorded as `auth-disabled`.** That is fine on a laptop. On a shared node
  it means any local account can spend the deployment's daily LLM budget, even
  with the port bound to `127.0.0.1`. Configure tokens before enabling
  `BAMBOO_REST_ENABLED` anywhere shared.

The token never reaches the browser. The monitor's Django backend holds it and
proxies; see [`rest-api.md`](rest-api.md#integrating-the-panda-monitor).

---

## Recommended Deployment Pattern (CERN / Internal)

1. Bind HTTP server to `127.0.0.1`
2. Place behind a firewall or reverse proxy if exposing externally
3. Store tokens in `/etc/bamboo/tokens.txt`
4. Use per-client tokens
5. Rotate tokens by:
   - Adding new token
   - Updating client
   - Removing old token

---

## CI Integration

GitHub Actions workflow runs:

- flake8
- pylint
- pydocstyle
- pyright

Dev dependencies are listed in:

```requirements-dev.txt```

Install locally with:

```pip install -r requirements-dev.txt```


---

## Development Mode

If neither `BAMBOO_MCP_TOKENS_FILE` nor `BAMBOO_MCP_TOKENS` is set:

- Authentication is disabled
- HTTP endpoint is open (intended for local development only)

---

## Superuser / Developer Mode

Bamboo supports an optional in-session developer tier that gates developer tools
(currently `pilot_code_query`) behind a password prompt in the UI. This is
separate from the Bearer token system — it does not affect what tools are
callable on the MCP server, only what the UI exposes.

### Enabling

Set `BAMBOO_SUPERUSER_PASSWORD` in `bamboo_env.sh`:

```bash
export BAMBOO_SUPERUSER_PASSWORD="yourpassword"
```

When this is set:
- **Streamlit:** a "Developer access" section appears in the sidebar with a password
  input and Unlock/Lock button.
- **TUI:** the `/superuser <password>` command becomes active.

When unset or empty, both UI features are hidden entirely.

### Password comparison

The password is compared using `hmac.compare_digest` (constant-time) to resist
timing attacks. The value is never logged, never sent to the MCP server, and
never transmitted over the network.

### Security scope

Superuser mode is a **UI-level gate only**. It controls which evidence panels are
shown in the Streamlit interface for developer tools — it does not restrict which
tools are callable by any MCP client, including Claude Desktop or the MCP
Inspector. For network-level access control, use Bearer token authentication
(described above).

This design is intentional: Bamboo targets internal scientific teams where tool
discoverability is more important than tool lockdown, but where visual clutter
for casual users should be minimised.

---

## Future Enhancements (Optional)

Possible future upgrades:

- Token hot-reload without restart
- Rate limiting
- IP allowlisting
- Integration with institutional SSO
- Structured audit logging per client_id

---

## Summary

Bamboo now supports:

- Per-client Bearer token authentication
- 256-bit secure token generation
- CI enforcement of style, linting, and type checking
- Dev-friendly defaults with secure production behavior

This keeps Bamboo lightweight while providing strong internal security.

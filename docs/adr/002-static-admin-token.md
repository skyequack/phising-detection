# ADR 002: One static bearer token for admin access

Status: accepted (2026-10-11)

## Context
Training a model and switching the served model must not be public. The system has no user accounts and no reason to
have them: one operator, one deployment. A login system would add a users table, password storage and session handling,
each a new way to be wrong.

## Decision
Admin routes require `Authorization: Bearer <ADMIN_TOKEN>`, where the token is an environment variable.
- The comparison uses `secrets.compare_digest`.
- Admin is disabled, with every call returning 401, unless the token is at least 16 characters and not the
  `.env.example` placeholder. A forgotten or weak token therefore fails closed instead of opening admin.
- The same 401 body is returned whatever the reason, and every denied attempt is audited.
- The token is read on each request, never logged, never stored in the database, and not allowed through CORS
  (`Authorization` is not an allowed header), so a page on another origin cannot use it.

## Consequences
- The audit log's actor is the label `admin`. It cannot say which person did something.
- Rotating the token means changing the variable and restarting the server.
- There is no rate limit on failed attempts, so a short but allowed token (16 characters from a small alphabet) could be
  guessed by brute force over the network. Use a random 32-byte token and keep the server off the open internet.
- Nothing here provides HTTPS. A bearer token over plain HTTP can be read by anyone on the path; use a reverse proxy
  for anything beyond localhost.
- Because the token travels in a header, not a cookie, cross-site request forgery does not apply.

## Alternatives considered
A user table with hashed passwords and sessions (more code and more risk for one operator); OAuth or an identity provider
(external dependency); IP allow-listing (fragile behind proxies). A later move to per-user accounts would need an
`actor` that is a real identity; the audit schema already stores one.

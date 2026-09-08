# Security Policy

## Reporting a vulnerability

Email **contact@pixeltable.com** with the details. Do not open a public issue, a discussion, or a
pull request for a suspected vulnerability.

Include what you have: the version (`pxt --version` or `pixeltable.__version__`), the platform, a
description of the issue, and the smallest reproduction you can manage. If you are unsure whether
something counts, send it anyway.

We will acknowledge your report and tell you whether we consider it a vulnerability and what we
plan to do. Please give us time to ship a fix before disclosing it publicly.

## Supported versions

Fixes land in the current release on PyPI. Pixeltable releases roughly weekly, so upgrading is
usually the fastest route to a fix:

```bash
pip install -U pixeltable
```

## What is in scope

Pixeltable runs several things worth naming, because they shape what a vulnerability looks like
here:

- **An embedded PostgreSQL server**, on a local socket under `$PIXELTABLE_HOME` (default
  `~/.pixeltable`). Reports about that data being readable or writable by an unintended local user
  are in scope.
- **User-supplied Python.** A UDF is your own code, and a computed column runs it. Pixeltable does
  not sandbox it, so a UDF doing something dangerous is not itself a vulnerability. A path by
  which *someone else's* code or data causes your process to execute code you did not write is.
- **HTTP endpoints** started by `pxt service update` or mounted through `FastAPIRouter`. These bind
  `127.0.0.1` by default. Exposing one on a public interface is a deployment decision; a route
  leaking data it was not declared to return, or accepting input that escapes its declared type,
  is in scope.
- **Media and file handling.** Pixeltable fetches and decodes images, video, audio, and documents
  from URLs and paths you give it. Reports about handling of untrusted media are in scope.
- **Credentials.** API keys come from the environment, `pixeltable.toml`, or Cloud secrets. A key
  appearing in a log, an error message, a stored column, or a `--json` payload is in scope.

## What is out of scope

- Vulnerabilities in third-party packages. Report those upstream; tell us if Pixeltable's usage
  makes an upstream issue materially worse.
- Anything requiring an attacker to already have the ability to run code as your user, or to write
  to your `$PIXELTABLE_HOME`.
- The absence of sandboxing around UDFs, which is by design and documented above.

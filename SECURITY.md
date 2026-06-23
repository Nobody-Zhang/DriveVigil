# Security Policy

## Supported versions

This repository is the competition code for the 18th Challenge Cup (Huawei Cloud
track). It is provided as a reference implementation and archive rather than an
actively maintained product. Security fixes, when feasible, are applied to the
`main` branch only.

| Version | Supported |
| ------- | --------- |
| `main`  | ✅        |
| Releases ≤ v1.0 | ⚠️ best effort |

## Reporting a vulnerability

Please **do not** open a public issue for security-sensitive reports.

- Preferred: use GitHub's **private vulnerability reporting**
  ([Security → Report a vulnerability](../../security/advisories/new)).
- Alternatively, email the maintainer: **imgongbozhang@gmail.com**.

Please include enough detail to reproduce (affected file/component, steps, and
impact). We will acknowledge receipt as soon as we reasonably can and keep you
informed of progress. Because this is archival competition code, response times
are best-effort.

## Credential handling

This project integrates with Huawei Cloud (ModelArts, OBS, SIS) and DashScope,
all of which require secrets. The project is designed so that **no credential
ever lives in source code**:

- All secrets are read at runtime from environment variables via a local `.env`
  file. The template is [`.env.example`](.env.example); the real `.env` is
  **gitignored** (along with `*.pem`, `*.key`, `*.secret`, `credentials.csv`).
- Relevant variables include `HUAWEICLOUD_AK` / `HUAWEICLOUD_SK`,
  `HUAWEICLOUD_TOKEN`, `HUAWEICLOUD_PROJECT_ID`, `HUAWEICLOUD_IMA_ID`, the
  `HUAWEICLOUD_*_URL` endpoints, and `DASHSCOPE_API_KEY`.

### If you ever leak a credential

1. **Rotate it immediately** in the Huawei Cloud / DashScope console (the leaked
   value must be considered compromised even if the commit is later removed).
2. Remove it from history (e.g. `git filter-repo`) and force-push if it was
   pushed.
3. Never paste live AK/SK or tokens into issues, PRs, or logs. Scrub logs before
   attaching them to a bug report.

## Scope notes

- Large binary assets (model weights, TensorRT engines) are distributed via
  GitHub Releases, not the git tree.
- Vendored SDKs under `apig_sdk/`, `edge/voice/huaweicloud-python-sdk-sis-1.8.1/`,
  and `edge/cloud_finetune/yolov5/` are upstream third-party code; vulnerabilities
  in them should be reported to their respective upstream projects.

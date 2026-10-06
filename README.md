# climateiq-cnn
ClimateIQ CNN Workstream

## Deployment

GitHub Actions deploys this repo's services; `climateiq-terraform` owns the infrastructure
around them (service accounts, IAM, buckets, triggers, memory, timeouts, env vars). CI only
updates code and images. Bumping the submodule in `climateiq-terraform` and applying still
recreates all 12 functions from the pinned code, which overwrites anything CI deployed since.

| Event | Target |
| --- | --- |
| Push to `main` (after `ci-ok`) | `climateiq-test`, services whose source changed since the last green CI run on `main` |
| PR `main` → `release`, then merge (after `ci-ok`) | `climateiq`, services changed since the last green CI run on `release` |
| Actions → **Deploy** (manual) | redeploy or roll back at a chosen commit; production only from `release` |

Covered: the 12 pipeline Cloud Functions (`.github/deploy/functions.json`) and the `add-area`
Cloud Run job image. The H3 functions and the AtmoML image are still deployed by hand.

One-time setup: create GitHub Environments `test` and `production`, each with the variable
`GCP_PROJECT_ID` (`climateiq-test` / `climateiq`) and the secret `GCP_SA_KEY` (a JSON key for
that project's Compute Engine default service account, which holds Editor; rotate it
periodically), restrict `production` deployments to the `release` branch, require `ci-ok` on `main`,
and create `release` from `main` as a protected branch (PRs only, `ci-ok` required, no force
pushes, restricted pushers). Creating the branch deploys nothing; run Actions → **Deploy** from
`release` with `functions: all` and `add_area` ticked for the first full production deploy.
Merge into `release` with a merge commit or fast-forward, never a squash, so it stays an
ancestor of `main`.

## Setup

Instructions for setting up dev environment.

### Install GCloud CLI

To install `gcloud` CLI on linux:

```sh
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
```

Then login using:

```sh
gcloud init
```

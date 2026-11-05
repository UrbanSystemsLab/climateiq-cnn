# climateiq-cnn
ClimateIQ CNN Workstream

## Setup

Instructions for setting up dev environment.

## Pipeline

[usl_pipeline/README.md](usl_pipeline/README.md) describes the stages, what each reads and writes, and where to change things.

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

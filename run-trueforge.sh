#!/usr/bin/env bash
# Start TrueForge in standalone (SQLite, no auth) mode on the GPU host.
# Bound to localhost only: standalone mode has no login and must not be exposed.
set -euo pipefail
export PATH="${PNPM_SHIM_DIR:-$PWD/bin}:$PATH"
# env-paths resolves the app data dir from XDG_DATA_HOME; without this the
# database lands in ~/.local/share instead of beside the project.
export XDG_DATA_HOME="${KERNEL_PREFLIGHT_DATA:-$PWD/.trueforge-data}"
# Only needed behind a TLS-intercepting proxy whose root CA is in the system
# store but not in Node's bundled one — model calls otherwise fail with
# UNABLE_TO_GET_ISSUER_CERT_LOCALLY. NODE_OPTIONS is hardcoded by the package
# script, so --use-system-ca would be discarded; this survives.
export NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-/etc/ssl/certs/ca-certificates.crt}"
export APP_DATA_DIR_SUFFIX=kernel-preflight
export PORT=8790
# Path to a TrueForge checkout; override for your layout.
cd "${TRUEFORGE_DIR:?set TRUEFORGE_DIR to your trueforge checkout}"
exec pnpm run standalone:dev:no-watch

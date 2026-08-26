#!/usr/bin/env bash
# Sync source to the GPU host.
#
# --delete and the pycache purge are both load-bearing. rsync -a preserves
# mtimes, so a synced .py can look *older* than the .pyc already on the far side
# and Python will keep running the cached bytecode. That produced a stale
# `unknown backend` error for a backend that was registered in both copies of the
# file, and it is the same failure mode as forgetting to restart the MCP server.
set -euo pipefail
HOST="${1:-bigdaddy.rycerz.es}"
REMOTE="/home/asyin/swapnil/trueforge-hack/kernel-preflight"
ssh -o BatchMode=yes "$HOST" "find $REMOTE/src $REMOTE/tests -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true"
rsync -az --delete src/ "$HOST:$REMOTE/src/"
rsync -az --delete tests/ "$HOST:$REMOTE/tests/"
rsync -az docker/ "$HOST:$REMOTE/docker/"
rsync -az benchmark/ "$HOST:$REMOTE/benchmark/"
echo "synced to $HOST"

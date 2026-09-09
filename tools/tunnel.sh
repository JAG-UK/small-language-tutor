#!/usr/bin/env bash
#
# Reach a tutor running on another machine, without opening a port on it.
#
#   tools/tunnel.sh jag@gpu-box            # then open http://localhost:5001
#   tools/tunnel.sh jag@gpu-box 5555       # if 5001 is busy here
#
# The far end keeps binding to its own loopback, so the port is never on the
# network: SSH carries the traffic and supplies the encryption that HTTP Basic
# does not. Leave this running; Ctrl-C closes the tunnel.
#
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
fi

remote="$1"
local_port="${2:-5001}"
remote_port="${SLT_REMOTE_PORT:-5001}"

echo "Forwarding localhost:${local_port} -> ${remote} (its localhost:${remote_port})" >&2
echo "Open http://localhost:${local_port} — Ctrl-C to close." >&2

# -N: no remote command, just the forward.
# ExitOnForwardFailure: fail loudly if the local port is taken, rather than
# appearing to work and forwarding nothing.
exec ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -L "${local_port}:127.0.0.1:${remote_port}" \
    "${remote}"

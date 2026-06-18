#!/usr/bin/env bash
# Open SSH local forwards for remote ForeAct servers.
#
# Defaults:
#   local 127.0.0.1:5100 -> remote 127.0.0.1:5100
#   local 127.0.0.1:5105 -> remote 127.0.0.1:5105
#   ...
#   local 127.0.0.1:5110 -> remote 127.0.0.1:5110
#
# Usage:
#   bash scripts/foreact_ssh_tunnel.sh
#
# Optional overrides:
#   REMOTE_HOST=115.190.6.133
#   REMOTE_USER=xiahongyu
#   REMOTE_BIND_HOST=127.0.0.1
#   LOCAL_BIND_HOST=127.0.0.1
#   PORTS="5100 5105 5106 5107 5108 5109 5110"
#   LOCAL_PORT_OFFSET=0
#   SSH_PORT=22
#
# Example when local ports are occupied:
#   LOCAL_PORT_OFFSET=10000 bash scripts/foreact_ssh_tunnel.sh
# This maps local 15100 -> remote 5100, local 15105 -> remote 5105, etc.
#
# Then point eval_gui.py ForeAct to:
#   host: 127.0.0.1
#   port: 5100, 5105, 5106, 5107, 5108, 5109, or 5110

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-115.190.6.133}"
REMOTE_USER="${REMOTE_USER:-xiahongyu}"
REMOTE_BIND_HOST="${REMOTE_BIND_HOST:-127.0.0.1}"
LOCAL_BIND_HOST="${LOCAL_BIND_HOST:-127.0.0.1}"
PORTS="${PORTS:-5100 5105 5106 5107 5108 5109 5110}"
LOCAL_PORT_OFFSET="${LOCAL_PORT_OFFSET:-0}"
SSH_PORT="${SSH_PORT:-22}"

if ! [[ "$LOCAL_PORT_OFFSET" =~ ^-?[0-9]+$ ]]; then
    echo "LOCAL_PORT_OFFSET must be an integer." >&2
    exit 2
fi

if ! [[ "$SSH_PORT" =~ ^[0-9]+$ ]]; then
    echo "SSH_PORT must be an integer." >&2
    exit 2
fi

ssh_args=(
    -N
    -T
    -p "$SSH_PORT"
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
)

echo "Opening ForeAct SSH tunnel to ${REMOTE_USER}@${REMOTE_HOST}:${SSH_PORT}"
echo "Mappings:"

for remote_port in $PORTS; do
    if ! [[ "$remote_port" =~ ^[0-9]+$ ]]; then
        echo "Invalid port in PORTS: ${remote_port}" >&2
        exit 2
    fi
    local_port=$((remote_port + LOCAL_PORT_OFFSET))
    if ((local_port < 1 || local_port > 65535 || remote_port < 1 || remote_port > 65535)); then
        echo "Port out of range: local=${local_port}, remote=${remote_port}" >&2
        exit 2
    fi
    forward="${LOCAL_BIND_HOST}:${local_port}:${REMOTE_BIND_HOST}:${remote_port}"
    ssh_args+=(-L "$forward")
    printf '  %s:%d -> %s:%d\n' "$LOCAL_BIND_HOST" "$local_port" "$REMOTE_BIND_HOST" "$remote_port"
done

echo
echo "Keep this process running while evaluating. Press Ctrl-C to close the tunnel."
exec ssh "${ssh_args[@]}" "${REMOTE_USER}@${REMOTE_HOST}"

#!/usr/bin/env bash
# [TAG_RPC_CACHE_ATOMIC] starts one rpc-server with the file cache enabled and runs
# test-rpc-cache against it: compute inputs must not be cached, weights must be, and a
# truncated cache entry must be refused instead of served
set -euo pipefail

server=$1
client=$2
port=$((40000 + $$ % 10000))
endpoint="127.0.0.1:${port}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_port() {
    local port=$1
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&-
            exec 3<&-
            return 0
        fi
        sleep 0.05
    done
    return 1
}

LLAMA_CACHE="$test_dir" "$server" -H 127.0.0.1 -p "$port" -c > "$test_dir/server.log" 2>&1 &
pid=$!
wait_for_port "$port"

"$client" "$endpoint" "$test_dir/rpc"

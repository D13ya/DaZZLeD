#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${ROOT_DIR}/api/proto/v1"

INCLUDES=("${PROTO_DIR}")
if [[ -n "${PROTOC_INCLUDE:-}" ]]; then
  INCLUDES+=("${PROTOC_INCLUDE}")
fi

PROTO_FLAGS=()
for inc in "${INCLUDES[@]}"; do
  PROTO_FLAGS+=("-I" "${inc}")
done

protoc \
  "${PROTO_FLAGS[@]}" \
  --go_out="${PROTO_DIR}" \
  --go_opt=paths=source_relative \
  --go-grpc_out="${PROTO_DIR}" \
  --go-grpc_opt=paths=source_relative \
  "${PROTO_DIR}/service.proto"

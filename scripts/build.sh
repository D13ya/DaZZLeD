#!/usr/bin/env bash
set -euo pipefail

mkdir -p bin
go build -o bin/client ./cmd/client
go build -o bin/server ./cmd/server
go build -o bin/setup ./cmd/setup

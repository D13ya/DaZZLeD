PROTO_DIR := api/proto/v1

.PHONY: gen-proto build build-client build-server build-setup

gen-proto:
	./scripts/gen_proto.sh

build: build-client build-server build-setup

build-client:
	go build -o bin/client ./cmd/client

build-server:
	go build -o bin/server ./cmd/server

build-setup:
	go build -o bin/setup ./cmd/setup

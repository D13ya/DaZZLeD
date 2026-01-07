Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$protoDir = Resolve-Path (Join-Path $PSScriptRoot "..\\api\\proto\\v1")
$includeDir = "C:/Program Files/protoc-win64/include"

protoc `
  -I "$protoDir" `
  -I "$includeDir" `
  --go_out="$protoDir" --go_opt=paths=source_relative `
  --go-grpc_out="$protoDir" --go-grpc_opt=paths=source_relative `
  "$protoDir\\service.proto"

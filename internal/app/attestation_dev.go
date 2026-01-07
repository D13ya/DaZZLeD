//go:build dev
// +build dev

package app

// DevAttestationSecret returns the well-known attestation secret for development.
// Both client and server use this same secret in dev mode.
// WARNING: This should NEVER be used in production!
var DevAttestationSecret = []byte("dazzled-dev-attestation-secret!")

// IsDevMode returns true in development builds.
func IsDevMode() bool {
	return true
}

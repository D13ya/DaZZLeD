//go:build !dev
// +build !dev

package app

// DevAttestationSecret is nil in production builds - attestation secrets must be provided.
var DevAttestationSecret []byte = nil

// IsDevMode returns false in production builds.
func IsDevMode() bool {
	return false
}

//go:build dev
// +build dev

package main

// In development builds, insecure mode is allowed for local testing.
// DO NOT use development builds in production environments.

func init() {
	productionMode = false
}

var productionMode bool

// IsInsecureAllowed returns true only in development builds.
func IsInsecureAllowed() bool {
	return true
}

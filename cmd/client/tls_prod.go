//go:build !dev
// +build !dev

package main

// In production builds, insecure mode is not available.
// The --insecure flag will be ignored and TLS is always required.

func init() {
	// Override the insecure flag behavior in production
	productionMode = true
}

var productionMode bool

// IsInsecureAllowed returns false in production builds.
func IsInsecureAllowed() bool {
	return false
}

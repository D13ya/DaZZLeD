//go:build !dev
// +build !dev

package main

import "testing"

// TestProdBuildInsecureNotAllowed verifies that production builds reject insecure mode.
// This test only compiles/runs WITHOUT the "dev" build tag (default).
func TestProdBuildInsecureNotAllowed(t *testing.T) {
	if IsInsecureAllowed() {
		t.Error("production build should NOT allow insecure mode")
	}
}

// TestProdBuildProductionModeTrue verifies production builds set productionMode = true.
func TestProdBuildProductionModeTrue(t *testing.T) {
	if !productionMode {
		t.Error("production build should have productionMode = true")
	}
}

// TestProdBuildInsecureCredsFail verifies buildCreds returns an error in insecure mode.
func TestProdBuildInsecureCredsFail(t *testing.T) {
	_, err := buildCreds("", "", "", true)
	if err == nil {
		t.Error("insecure mode should error in production build")
	}
}

// TestProdBuildTLSRequired verifies production builds require TLS configuration.
func TestProdBuildTLSRequired(t *testing.T) {
	// Without insecure flag and without CA, should require TLS
	_, err := buildCreds("", "", "", false)
	if err == nil {
		t.Error("production build should require TLS CA when not insecure")
	}
}

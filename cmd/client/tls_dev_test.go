//go:build dev
// +build dev

package main

import "testing"

// TestDevBuildInsecureAllowed verifies that dev builds allow insecure mode.
// This test only compiles/runs with the "dev" build tag: go test -tags=dev
func TestDevBuildInsecureAllowed(t *testing.T) {
	if !IsInsecureAllowed() {
		t.Error("dev build should allow insecure mode")
	}
}

// TestDevBuildProductionModeFalse verifies dev builds set productionMode = false.
func TestDevBuildProductionModeFalse(t *testing.T) {
	if productionMode {
		t.Error("dev build should have productionMode = false")
	}
}

// TestDevBuildInsecureCredsSucceed verifies buildCreds returns valid creds in insecure mode.
func TestDevBuildInsecureCredsSucceed(t *testing.T) {
	creds, err := buildCreds("", "", "", true)
	if err != nil {
		t.Errorf("insecure mode should not error in dev build: %v", err)
	}
	if creds == nil {
		t.Error("expected non-nil insecure credentials")
	}
}

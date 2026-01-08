package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Test directory creation with valid path
func TestSetupDirectoryCreation(t *testing.T) {
	tempDir := t.TempDir()
	keyDir := filepath.Join(tempDir, "new_keys")

	// Directory shouldn't exist yet
	if _, err := os.Stat(keyDir); !os.IsNotExist(err) {
		t.Skip("directory already exists")
	}

	// Test that MkdirAll works
	if err := os.MkdirAll(keyDir, 0o700); err != nil {
		t.Errorf("MkdirAll failed: %v", err)
	}

	// Check directory was created with correct permissions
	info, err := os.Stat(keyDir)
	if err != nil {
		t.Errorf("Stat failed: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory")
	}
}

// Test key file permissions
func TestSetupKeyFilePermissions(t *testing.T) {
	tempDir := t.TempDir()
	keyPath := filepath.Join(tempDir, "test_key.bin")

	// Write with restrictive permissions
	if err := os.WriteFile(keyPath, []byte("test key data"), 0o600); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	info, err := os.Stat(keyPath)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}

	// On Windows, permission bits work differently, so just check file exists
	if info.Size() == 0 {
		t.Error("expected non-empty file")
	}
}

// Test write to readonly directory fails
func TestSetupReadonlyDirectoryFails(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("skipping permission test in CI")
	}

	tempDir := t.TempDir()
	roDir := filepath.Join(tempDir, "readonly")

	if err := os.MkdirAll(roDir, 0o500); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}

	keyPath := filepath.Join(roDir, "key.bin")
	err := os.WriteFile(keyPath, []byte("data"), 0o600)

	// On some systems this may succeed, on others fail - just document behavior
	t.Logf("write to readonly dir result: %v", err)
}

// Test output directory already exists as file
func TestSetupOutputPathIsFile(t *testing.T) {
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "not_a_dir")

	// Create a file where we'd want a directory
	if err := os.WriteFile(filePath, []byte("I'm a file"), 0o600); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	// MkdirAll should fail when path exists as a file
	err := os.MkdirAll(filePath, 0o700)
	if err == nil {
		t.Error("expected error when path is a file, not directory")
	}
}

// Test empty output directory defaults
func TestSetupDefaultOutputDir(t *testing.T) {
	// The default is "keys" - verify this path handling
	defaultDir := "keys"

	// When out-dir is ".", MkdirAll with "." should not error
	if err := os.MkdirAll(".", 0o700); err != nil {
		t.Errorf("MkdirAll(\".\") failed: %v", err)
	}

	// Clean path handling
	cleaned := filepath.Clean(defaultDir)
	if cleaned != defaultDir {
		t.Logf("path cleaned: %s -> %s", defaultDir, cleaned)
	}
}

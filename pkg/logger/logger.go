package logger

import (
	"log"
	"os"
)

// New returns a standard logger with a consistent prefix.
func New(prefix string) *log.Logger {
	return log.New(os.Stdout, prefix, log.LstdFlags|log.LUTC)
}

package profiler

import "time"

// Timer is a lightweight timing helper for instrumentation.
type Timer struct {
	start time.Time
}

func Start() Timer {
	return Timer{start: time.Now()}
}

func (t Timer) Elapsed() time.Duration {
	return time.Since(t.start)
}

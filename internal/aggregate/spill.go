// Package aggregate provides a growable buffer that starts in memory and
// spills to a randomly named temporary file once a threshold is crossed.
//
// The soft cache limit is used only to sweep completed/stale spill files. It
// is never applied to a buffer that is still being written, so a single large
// response is always returned in full.
package aggregate

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	tempPrefix = "agg-"
	tempSuffix = ".tmp"

	// Each Store writes into its own discoverable instance directory under a
	// fixed base directory. The directory itself is locked for the lifetime of
	// the instance, allowing other instances to distinguish active directories
	// from directories left behind by a killed process.
	instancePrefix = "run-"

	defaultBaseName = "oai-to-gemini-sse"
	legacyPrefix    = "oai-to-gemini-sse-"

	// Newly created spill files are not eligible for sweeping. This also
	// protects against a snapshot/registration race between concurrent
	// buffers. Files are deleted by Buffer.Close as soon as a request ends.
	sweepGracePeriod = time.Minute
	sweepInterval    = time.Minute
)

// Store owns one instance directory under a base cache directory and tracks
// files that are currently being written. Sweep only removes files that are
// not active. Multiple instances may share a base directory because each one
// writes into a separate instance directory guarded by an OS file lock.
type Store struct {
	dir     string
	baseDir string
	lock    *os.File
	// removeBaseOnClose is set only for the default base directory, which is
	// created by this process and can be removed when it becomes empty.
	removeBaseOnClose bool

	threshold int64
	maxBytes  int64
	logger    *log.Logger

	mu             sync.Mutex
	active         map[string]struct{}
	spilledBuffers int
	closed         bool
	finalized      bool
	closeErr       error
	stop           chan struct{}
	done           chan struct{}
	closeOnce      sync.Once
}

func NewStore(dir string, threshold, maxBytes int64, logger *log.Logger) (*Store, error) {
	if threshold <= 0 {
		threshold = 1 << 20
	}
	if maxBytes <= 0 {
		maxBytes = 64 << 20
	}
	if logger == nil {
		logger = log.New(io.Discard, "", 0)
	}

	defaultBase := false
	if dir == "" {
		dir = filepath.Join(os.TempDir(), defaultBaseName)
		defaultBase = true
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("create sse cache dir %q: %w", dir, err)
	}
	if defaultBase {
		cleanLegacyDefaultDirs(dir, logger)
	}

	instanceDir, err := os.MkdirTemp(dir, instancePrefix+"*")
	if err != nil {
		return nil, fmt.Errorf("create sse cache instance dir under %q: %w", dir, err)
	}
	lock, err := os.Open(instanceDir)
	if err != nil {
		_ = os.RemoveAll(instanceDir)
		return nil, fmt.Errorf("open sse cache instance dir for locking: %w", err)
	}
	if err := lockFile(lock); err != nil {
		_ = lock.Close()
		_ = os.RemoveAll(instanceDir)
		return nil, fmt.Errorf("lock sse cache instance dir: %w", err)
	}

	s := &Store{
		dir:               instanceDir,
		baseDir:           dir,
		lock:              lock,
		removeBaseOnClose: defaultBase,
		threshold:         threshold,
		maxBytes:          maxBytes,
		logger:            logger,
		active:            make(map[string]struct{}),
		stop:              make(chan struct{}),
		done:              make(chan struct{}),
	}
	s.Sweep()
	go s.cleanupLoop()
	return s, nil
}

func lockFile(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
}

func unlockFile(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}

// cleanLegacyDefaultDirs removes old top-level oai-to-gemini-sse-* directories
// created by previous versions in the system temp directory. Only directories
// whose contents are entirely stale spill files in this service's format are
// removed; anything else is left untouched.
func cleanLegacyDefaultDirs(base string, logger *log.Logger) {
	parent := filepath.Dir(base)
	entries, err := os.ReadDir(parent)
	if err != nil {
		logger.Printf("sse cache legacy sweep: read dir: %v", err)
		return
	}
	cutoff := time.Now().Add(-sweepGracePeriod)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, legacyPrefix) {
			continue
		}
		path := filepath.Join(parent, name)
		info, err := entry.Info()
		if err != nil {
			continue
		}
		contents, err := os.ReadDir(path)
		if err != nil {
			continue
		}
		if len(contents) == 0 && info.ModTime().After(cutoff) {
			continue
		}
		if !removeStaleSpillFiles(path, cutoff, logger) {
			continue
		}
		if err := os.Remove(path); err == nil {
			logger.Printf("sse cache legacy sweep removed %s", name)
		}
	}
}

// onlySpillFiles reports whether dir is empty or contains only files in this
// service's spill format. It is used before removing a whole instance dir so
// unrelated files are never deleted.
func onlySpillFiles(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if entry.IsDir() {
			return false
		}
		name := entry.Name()
		if !strings.HasPrefix(name, tempPrefix) || !strings.HasSuffix(name, tempSuffix) {
			return false
		}
	}
	return true
}

// removeStaleSpillFiles deletes spill files in dir when every entry is a stale
// file in this service's format. It returns true only when the directory is
// empty afterwards, so the caller can remove it.
func removeStaleSpillFiles(dir string, cutoff time.Time, logger *log.Logger) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if entry.IsDir() {
			return false
		}
		name := entry.Name()
		if !strings.HasPrefix(name, tempPrefix) || !strings.HasSuffix(name, tempSuffix) {
			return false
		}
		info, err := entry.Info()
		if err != nil || info.ModTime().After(cutoff) {
			return false
		}
	}
	for _, entry := range entries {
		path := filepath.Join(dir, entry.Name())
		if err := os.Remove(path); err != nil {
			return false
		}
		logger.Printf("sse cache legacy sweep removed %s", filepath.Base(path))
	}
	return true
}

func (s *Store) cleanupLoop() {
	defer close(s.done)
	ticker := time.NewTicker(sweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.Sweep()
		case <-s.stop:
			return
		}
	}
}

func (s *Store) Dir() string     { return s.dir }
func (s *Store) MaxBytes() int64 { return s.maxBytes }

func (s *Store) NewBuffer() *Buffer {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return &Buffer{store: s, threshold: s.threshold, closed: true}
	}
	return &Buffer{store: s, threshold: s.threshold}
}

func (s *Store) register(path string) {
	s.mu.Lock()
	s.active[path] = struct{}{}
	s.mu.Unlock()
}

func (s *Store) unregister(path string) {
	s.mu.Lock()
	delete(s.active, path)
	s.mu.Unlock()
}

// Sweep removes completed spill files, oldest first, until the total size of
// completed files is within the soft limit. Active files are never touched.
// It also removes stale instance directories left behind by killed processes,
// but skips directories whose lock is still held by another live instance.
func (s *Store) Sweep() {
	s.sweepSpillFiles(s.dir)
	if s.baseDir != s.dir {
		// Older versions wrote spill files directly into the configured
		// directory. Sweep those too, without touching instance subdirectories.
		s.sweepSpillFiles(s.baseDir)
	}
	s.sweepStaleInstanceDirs()
}

func (s *Store) sweepSpillFiles(dir string) {
	s.mu.Lock()
	active := make(map[string]struct{}, len(s.active))
	for p := range s.active {
		active[p] = struct{}{}
	}
	s.mu.Unlock()

	entries, err := os.ReadDir(dir)
	if err != nil {
		s.logger.Printf("sse cache sweep: read dir: %v", err)
		return
	}

	type cachedFile struct {
		path    string
		size    int64
		modTime time.Time
	}
	var files []cachedFile
	var total int64

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, tempPrefix) || !strings.HasSuffix(name, tempSuffix) {
			continue
		}
		path := filepath.Join(dir, name)
		if _, ok := active[path]; ok {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if info.ModTime().After(time.Now().Add(-sweepGracePeriod)) {
			continue
		}
		files = append(files, cachedFile{path: path, size: info.Size(), modTime: info.ModTime()})
		total += info.Size()
	}

	if total <= s.maxBytes {
		return
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime.Before(files[j].modTime)
	})
	for _, f := range files {
		if total <= s.maxBytes {
			break
		}
		if err := os.Remove(f.path); err == nil {
			total -= f.size
			s.logger.Printf("sse cache sweep removed %s (%d bytes)", filepath.Base(f.path), f.size)
		}
	}
}

func (s *Store) sweepStaleInstanceDirs() {
	entries, err := os.ReadDir(s.baseDir)
	if err != nil {
		s.logger.Printf("sse cache sweep: read base dir: %v", err)
		return
	}
	own := filepath.Base(s.dir)
	cutoff := time.Now().Add(-sweepGracePeriod)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if name == own || !strings.HasPrefix(name, instancePrefix) {
			continue
		}
		path := filepath.Join(s.baseDir, name)
		info, err := entry.Info()
		if err != nil || info.ModTime().After(cutoff) {
			continue
		}

		lf, err := os.Open(path)
		if err != nil {
			continue
		}
		locked := lockFile(lf) == nil
		if !locked {
			_ = lf.Close()
			continue
		}
		if !onlySpillFiles(path) {
			_ = unlockFile(lf)
			_ = lf.Close()
			continue
		}
		removeErr := os.RemoveAll(path)
		_ = unlockFile(lf)
		_ = lf.Close()
		if removeErr == nil {
			s.logger.Printf("sse cache sweep removed stale instance dir %s", name)
		}
	}
}

func (s *Store) Close() error {
	s.closeOnce.Do(func() {
		close(s.stop)
		<-s.done

		// Do not remove the instance directory while a buffer still owns a
		// spilled file. Closing the store is the final lifecycle step:
		// handlers must have released their buffers first. New spills are
		// rejected from this point on, and in-memory buffers never touch the
		// directory. If a spilled buffer is still active, the last Buffer.Close
		// finalizes the removal.
		s.mu.Lock()
		s.closed = true
		s.mu.Unlock()
		s.finalizeIfIdle()
	})
	s.mu.Lock()
	err := s.closeErr
	s.mu.Unlock()
	return err
}

func (s *Store) finalizeIfIdle() {
	s.mu.Lock()
	if !s.closed || s.spilledBuffers > 0 || s.finalized {
		s.mu.Unlock()
		return
	}
	s.finalized = true
	s.mu.Unlock()

	var firstErr error
	if err := os.RemoveAll(s.dir); err != nil {
		firstErr = err
	}
	if s.lock != nil {
		if err := unlockFile(s.lock); err != nil && firstErr == nil {
			firstErr = err
		}
		if err := s.lock.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		s.lock = nil
	}
	if s.removeBaseOnClose {
		_ = os.Remove(s.baseDir)
	}

	s.mu.Lock()
	if firstErr != nil {
		s.closeErr = firstErr
	}
	s.mu.Unlock()
}

type nopCloser struct{}

func (nopCloser) Close() error { return nil }

// Buffer keeps data in memory until threshold is exceeded, then moves the
// already-written bytes into a temp file and appends there.
type Buffer struct {
	store     *Store
	threshold int64

	mu     sync.Mutex
	mem    bytes.Buffer
	file   *os.File
	path   string
	size   int64
	closed bool
}

func (b *Buffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return 0, os.ErrClosed
	}
	if err := b.prepareLocked(int64(len(p))); err != nil {
		return 0, err
	}
	if b.file != nil {
		n, err := b.file.Write(p)
		b.size += int64(n)
		return n, err
	}
	n, err := b.mem.Write(p)
	b.size += int64(n)
	return n, err
}

// WriteString is Write without converting s to a byte slice first.
func (b *Buffer) WriteString(s string) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return 0, os.ErrClosed
	}
	if err := b.prepareLocked(int64(len(s))); err != nil {
		return 0, err
	}
	if b.file != nil {
		n, err := b.file.WriteString(s)
		b.size += int64(n)
		return n, err
	}
	n, err := b.mem.WriteString(s)
	b.size += int64(n)
	return n, err
}

func (b *Buffer) prepareLocked(incoming int64) error {
	if b.file != nil || b.size+incoming <= b.threshold {
		return nil
	}
	if err := b.spillLocked(); err != nil {
		// Falling back to unbounded in-memory growth would defeat the spill
		// mechanism. Fail the aggregation instead so the caller can stop
		// reading upstream and return an error.
		return fmt.Errorf("spill sse data to file: %w", err)
	}
	return nil
}

func (b *Buffer) spillLocked() error {
	// Count the spill before creating the file so Store.Close cannot remove
	// the instance directory between creation and registration. If the store
	// is already closing, reject the spill instead of creating an orphan.
	b.store.mu.Lock()
	if b.store.closed {
		b.store.mu.Unlock()
		return os.ErrClosed
	}
	b.store.spilledBuffers++
	b.store.mu.Unlock()

	release := func() {
		b.store.mu.Lock()
		b.store.spilledBuffers--
		b.store.mu.Unlock()
	}

	f, err := os.CreateTemp(b.store.dir, tempPrefix+"*"+tempSuffix)
	if err != nil {
		release()
		return err
	}
	path := f.Name()
	// Register before writing so a concurrent sweep can never mistake this
	// file for an abandoned cache entry.
	b.store.register(path)
	if _, err := f.Write(b.mem.Bytes()); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		b.store.unregister(path)
		release()
		return err
	}
	b.mem.Reset()
	b.file = f
	b.path = path
	return nil
}

// Bytes returns a copy of the full buffer contents. It never truncates.
func (b *Buffer) Bytes() ([]byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.file == nil {
		out := make([]byte, b.mem.Len())
		copy(out, b.mem.Bytes())
		return out, nil
	}
	if err := b.file.Sync(); err != nil {
		return nil, err
	}
	return os.ReadFile(b.path)
}

// Reader returns an independent reader over the full buffer contents. Spilled
// data is read from the temp file rather than being loaded into memory.
func (b *Buffer) Reader() (io.ReadCloser, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.file == nil {
		data := make([]byte, b.mem.Len())
		copy(data, b.mem.Bytes())
		return io.NopCloser(bytes.NewReader(data)), nil
	}
	if err := b.file.Sync(); err != nil {
		return nil, err
	}
	return os.Open(b.path)
}

// ReaderAt returns a random-access reader over the full buffer contents and a
// closer for any resources it owns. Spilled data is read from the temp file;
// in-memory data is exposed without loading a spilled file into memory.
func (b *Buffer) ReaderAt() (io.ReaderAt, io.Closer, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return nil, nil, os.ErrClosed
	}
	if b.file == nil {
		return bytes.NewReader(b.mem.Bytes()), nopCloser{}, nil
	}
	if err := b.file.Sync(); err != nil {
		return nil, nil, err
	}
	f, err := os.Open(b.path)
	if err != nil {
		return nil, nil, err
	}
	return f, f, nil
}

func (b *Buffer) String() (string, error) {
	data, err := b.Bytes()
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (b *Buffer) Size() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.size
}

func (b *Buffer) Spilled() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.file != nil
}

func (b *Buffer) Path() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.path
}

// Close deletes the temp file, if any. It is safe to call more than once.
func (b *Buffer) Close() error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.closed = true
	f := b.file
	path := b.path
	b.file = nil
	b.mu.Unlock()

	var firstErr error
	if f != nil {
		if err := f.Close(); err != nil {
			firstErr = err
		}
	}
	if path != "" {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) && firstErr == nil {
			firstErr = err
		}
		b.store.unregister(path)
	}
	b.store.Sweep()

	if path != "" {
		b.store.mu.Lock()
		b.store.spilledBuffers--
		b.store.mu.Unlock()
		b.store.finalizeIfIdle()
	}
	return firstErr
}

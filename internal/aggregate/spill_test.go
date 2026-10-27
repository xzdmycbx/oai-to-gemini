package aggregate

import (
	"bytes"
	"crypto/sha256"
	"io"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const DefaultSoftLimitForTest int64 = 64 << 20

func testStore(t *testing.T, threshold, maxBytes int64) *Store {
	t.Helper()
	store, err := NewStore(t.TempDir(), threshold, maxBytes, log.New(io.Discard, "", 0))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	return store
}

func TestSpillToFileKeepsFullContent(t *testing.T) {
	store := testStore(t, 8, 1<<20)
	defer store.Close()

	buf := store.NewBuffer()
	for _, chunk := range []string{"hello", " ", "world"} {
		if _, err := buf.Write([]byte(chunk)); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
	if !buf.Spilled() {
		t.Fatal("expected buffer to spill to a file")
	}
	path := buf.Path()
	if path == "" {
		t.Fatal("expected temp file path")
	}

	got, err := buf.String()
	if err != nil {
		t.Fatalf("String: %v", err)
	}
	if got != "hello world" {
		t.Fatalf("content = %q, want %q", got, "hello world")
	}

	onDisk, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(onDisk) != got {
		t.Fatalf("file content = %q, want %q", onDisk, got)
	}

	if err := buf.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("temp file still exists: %v", err)
	}
}

func TestSoftLimitDoesNotTruncateActiveAggregation(t *testing.T) {
	store := testStore(t, 4, 8)
	defer store.Close()

	buf := store.NewBuffer()
	payload := bytes.Repeat([]byte("x"), 100)
	if _, err := buf.Write(payload); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if buf.Size() != 100 {
		t.Fatalf("size = %d, want 100", buf.Size())
	}
	if !buf.Spilled() {
		t.Fatal("expected spill")
	}

	// The soft limit is smaller than the active response, but sweeping must not
	// touch the active file or truncate the response.
	store.Sweep()
	if _, err := os.Stat(buf.Path()); err != nil {
		t.Fatalf("active spill file was removed by sweep: %v", err)
	}
	got, err := buf.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("content length = %d, want %d (must not truncate)", len(got), len(payload))
	}

	path := buf.Path()
	if err := buf.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("temp file still exists after close: %v", err)
	}
}

func TestSweepEvictsCompletedFilesOverSoftLimit(t *testing.T) {
	dir := t.TempDir()
	store, err := NewStore(dir, 1024, 10, log.New(io.Discard, "", 0))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer store.Close()

	now := time.Now()
	files := []struct {
		name string
		age  time.Duration
	}{
		{"agg-old.tmp", 3 * time.Hour},
		{"agg-mid.tmp", 2 * time.Hour},
		{"agg-new.tmp", 1 * time.Hour},
	}
	for _, f := range files {
		path := filepath.Join(store.Dir(), f.name)
		if err := os.WriteFile(path, bytes.Repeat([]byte("y"), 8), 0o600); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		ts := now.Add(-f.age)
		if err := os.Chtimes(path, ts, ts); err != nil {
			t.Fatalf("Chtimes: %v", err)
		}
	}

	store.Sweep()

	var total int64
	for _, f := range files {
		info, err := os.Stat(filepath.Join(store.Dir(), f.name))
		if err == nil {
			total += info.Size()
		}
	}
	if total > 10 {
		t.Fatalf("completed cache total = %d, want <= 10", total)
	}
	if _, err := os.Stat(filepath.Join(store.Dir(), "agg-new.tmp")); err != nil {
		t.Fatalf("newest file should be kept: %v", err)
	}
	if _, err := os.Stat(filepath.Join(store.Dir(), "agg-old.tmp")); !os.IsNotExist(err) {
		t.Fatalf("oldest file should be evicted: %v", err)
	}
}

func TestSpillFailureReturnsErrorInsteadOfGrowingInMemory(t *testing.T) {
	store := testStore(t, 8, 1<<20)
	defer store.Close()

	if err := os.RemoveAll(store.Dir()); err != nil {
		t.Fatalf("RemoveAll: %v", err)
	}

	buf := store.NewBuffer()
	if _, err := buf.Write(bytes.Repeat([]byte("x"), 100)); err == nil {
		t.Fatal("expected spill failure to be returned")
	}
	if buf.Size() != 0 {
		t.Fatalf("size = %d, want 0 after spill failure", buf.Size())
	}
}

func TestStoreCloseDoesNotRemoveActiveBuffers(t *testing.T) {
	store := testStore(t, 1, 1<<20)
	dir := store.Dir()

	buf := store.NewBuffer()
	if _, err := buf.Write(bytes.Repeat([]byte("x"), 100)); err != nil {
		t.Fatalf("Write: %v", err)
	}
	path := buf.Path()
	if path == "" {
		t.Fatal("expected spilled file")
	}

	if err := store.Close(); err != nil {
		t.Fatalf("Store.Close: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("active spill file was removed by Store.Close: %v", err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("active instance dir was removed by Store.Close: %v", err)
	}

	if err := buf.Close(); err != nil {
		t.Fatalf("Buffer.Close: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("spill file still exists after Buffer.Close: %v", err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("instance dir still exists after the last buffer closed: %v", err)
	}
}

func TestResponseLargerThanDefaultSoftLimitIsNotTruncated(t *testing.T) {
	store := testStore(t, 1<<20, DefaultSoftLimitForTest)
	defer store.Close()

	buf := store.NewBuffer()
	wantHash := sha256.New()
	chunk := make([]byte, 1<<20)
	const chunks = 65
	for i := 0; i < chunks; i++ {
		for j := range chunk {
			chunk[j] = byte((i*31+j*7)%251) + 1
		}
		if _, err := buf.Write(chunk); err != nil {
			t.Fatalf("Write %d: %v", i, err)
		}
		if _, err := wantHash.Write(chunk); err != nil {
			t.Fatalf("hash %d: %v", i, err)
		}
	}
	if !buf.Spilled() {
		t.Fatal("expected buffer to spill")
	}
	want := int64(chunks) << 20
	if buf.Size() != want {
		t.Fatalf("size = %d, want %d", buf.Size(), want)
	}
	reader, err := buf.Reader()
	if err != nil {
		t.Fatalf("Reader: %v", err)
	}
	gotHash := sha256.New()
	n, err := io.Copy(gotHash, reader)
	if err != nil {
		t.Fatalf("Copy: %v", err)
	}
	_ = reader.Close()
	if n != want {
		t.Fatalf("read %d bytes, want %d (must not truncate)", n, want)
	}
	if !bytes.Equal(gotHash.Sum(nil), wantHash.Sum(nil)) {
		t.Fatal("read content hash differs from written content")
	}
	path := buf.Path()
	if err := buf.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("temp file still exists after close: %v", err)
	}
}

func TestDefaultStoreCleansStaleLegacyDirs(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)

	oldDir := filepath.Join(tmp, legacyPrefix+"old")
	if err := os.Mkdir(oldDir, 0o700); err != nil {
		t.Fatal(err)
	}
	oldFile := filepath.Join(oldDir, "agg-old.tmp")
	if err := os.WriteFile(oldFile, bytes.Repeat([]byte("x"), 16), 0o600); err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(oldFile, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(oldDir, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	recentDir := filepath.Join(tmp, legacyPrefix+"recent")
	if err := os.Mkdir(recentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	recentFile := filepath.Join(recentDir, "agg-recent.tmp")
	if err := os.WriteFile(recentFile, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	unknownDir := filepath.Join(tmp, legacyPrefix+"unknown")
	if err := os.Mkdir(unknownDir, 0o700); err != nil {
		t.Fatal(err)
	}
	unknownFile := filepath.Join(unknownDir, "notes.txt")
	if err := os.WriteFile(unknownFile, []byte("keep me"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(unknownFile, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(unknownDir, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	store, err := NewStore("", 8, 1<<20, log.New(io.Discard, "", 0))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer store.Close()

	if _, err := os.Stat(oldDir); !os.IsNotExist(err) {
		t.Fatalf("stale legacy dir was not removed: %v", err)
	}
	if _, err := os.Stat(recentDir); err != nil {
		t.Fatalf("recent legacy dir should be kept: %v", err)
	}
	if _, err := os.Stat(unknownDir); err != nil {
		t.Fatalf("dir with non-service files should be kept: %v", err)
	}
}

func TestSharedBaseDirKeepsOtherInstanceActiveFiles(t *testing.T) {
	base := t.TempDir()
	logger := log.New(io.Discard, "", 0)

	a, err := NewStore(base, 1, 1<<20, logger)
	if err != nil {
		t.Fatalf("NewStore A: %v", err)
	}
	defer a.Close()
	b, err := NewStore(base, 1, 1<<20, logger)
	if err != nil {
		t.Fatalf("NewStore B: %v", err)
	}
	defer b.Close()

	if a.Dir() == b.Dir() {
		t.Fatal("instances sharing a base dir must get separate instance dirs")
	}

	buf := a.NewBuffer()
	payload := bytes.Repeat([]byte("x"), 100)
	if _, err := buf.Write(payload); err != nil {
		t.Fatalf("Write: %v", err)
	}
	path := buf.Path()
	if path == "" {
		t.Fatal("expected spilled file")
	}

	oldTime := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(path, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(a.Dir(), oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	b.Sweep()

	reader, err := buf.Reader()
	if err != nil {
		t.Fatalf("Reader after other instance sweep: %v", err)
	}
	got, err := io.ReadAll(reader)
	_ = reader.Close()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("active file content changed: got %d bytes, want %d", len(got), len(payload))
	}
}

func TestSweepRemovesStaleInstanceDirWithReleasedLock(t *testing.T) {
	base := t.TempDir()
	staleDir := filepath.Join(base, instancePrefix+"stale")
	if err := os.Mkdir(staleDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staleDir, "agg-stale.tmp"), []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(filepath.Join(staleDir, "agg-stale.tmp"), oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(staleDir, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	store, err := NewStore(base, 8, 1<<20, log.New(io.Discard, "", 0))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer store.Close()
	store.Sweep()

	if _, err := os.Stat(staleDir); !os.IsNotExist(err) {
		t.Fatalf("stale instance dir was not removed: %v", err)
	}
}

func TestSweepKeepsStaleInstanceDirWithUnknownFiles(t *testing.T) {
	base := t.TempDir()
	staleDir := filepath.Join(base, instancePrefix+"unknown")
	if err := os.Mkdir(staleDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staleDir, "notes.txt"), []byte("keep me"), 0o600); err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(filepath.Join(staleDir, "notes.txt"), oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(staleDir, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	store, err := NewStore(base, 8, 1<<20, log.New(io.Discard, "", 0))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer store.Close()
	store.Sweep()

	if _, err := os.Stat(staleDir); err != nil {
		t.Fatalf("instance dir with unknown files should be kept: %v", err)
	}
}

func TestSweepCleansLegacyFilesDirectlyInBaseDir(t *testing.T) {
	base := t.TempDir()
	legacy := filepath.Join(base, "agg-legacy.tmp")
	if err := os.WriteFile(legacy, bytes.Repeat([]byte("x"), 16), 0o600); err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(legacy, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	store, err := NewStore(base, 8, 10, log.New(io.Discard, "", 0))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer store.Close()
	store.Sweep()

	if _, err := os.Stat(legacy); !os.IsNotExist(err) {
		t.Fatalf("legacy base-dir spill file was not removed: %v", err)
	}
}

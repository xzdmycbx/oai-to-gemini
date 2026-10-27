package convert

import (
	"bytes"
	"io"
	"strings"
	"testing"
	"unicode/utf8"
)

type countingWriter struct {
	bytes.Buffer
	writes int
}

func (w *countingWriter) Write(p []byte) (int, error) {
	w.writes++
	return w.Buffer.Write(p)
}

type chunkReader struct {
	chunks     [][]byte
	index      int
	offset     int
	beforeRead func(index int)
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if r.index >= len(r.chunks) {
		return 0, io.EOF
	}
	if r.offset == 0 && r.beforeRead != nil {
		r.beforeRead(r.index)
	}
	chunk := r.chunks[r.index]
	n := copy(p, chunk[r.offset:])
	r.offset += n
	if r.offset >= len(chunk) {
		r.index++
		r.offset = 0
	}
	return n, nil
}

func TestWriteEscapedJSONStringBodyFromReaderDoesNotBufferInvalidUTF8(t *testing.T) {
	large := bytes.Repeat([]byte("a"), 1<<20)
	var out countingWriter
	r := &chunkReader{
		chunks: [][]byte{{0xff}, large},
		beforeRead: func(index int) {
			if index == 1 && out.writes == 0 {
				t.Error("output was not written before the large payload arrived")
			}
		},
	}
	if err := writeEscapedJSONStringBodyFromReader(&out, r); err != nil {
		t.Fatalf("writeEscapedJSONStringBodyFromReader: %v", err)
	}
	if !utf8.Valid(out.Bytes()) {
		t.Fatal("output is not valid UTF-8")
	}
	if bytes.Contains(out.Bytes(), []byte{0xff}) {
		t.Fatal("output contains the raw invalid byte")
	}
	if !bytes.Contains(out.Bytes(), []byte(strings.Repeat("a", 1024))) {
		t.Fatal("output is missing the large payload")
	}
}

func TestWriteEscapedJSONStringBodyFromReaderPreservesSplitRune(t *testing.T) {
	var out countingWriter
	r := &chunkReader{chunks: [][]byte{{0xe4}, {0xbd}, {0xa0}}}
	if err := writeEscapedJSONStringBodyFromReader(&out, r); err != nil {
		t.Fatalf("writeEscapedJSONStringBodyFromReader: %v", err)
	}
	if got := out.String(); got != "你" {
		t.Fatalf("output = %q, want %q", got, "你")
	}
}

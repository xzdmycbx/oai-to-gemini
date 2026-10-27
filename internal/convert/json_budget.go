package convert

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The raw request/response byte limits bound payload size, but they do not
// bound the number of Go values created by decoding shallow arrays. A small
// JSON null expands to a full struct element and a slice growth step, so a
// wide array can amplify memory far beyond its serialized size. These limits
// are checked by a lightweight structural scan before json.Unmarshal allocates
// the destination slices.
const (
	maxRequestTopLevelFields     = 4096
	maxRequestGenerationConfig   = 4096
	maxRequestMessages           = 65536
	maxRequestContentPartsPerMsg = 131072
	maxRequestContentPartsTotal  = 1 << 20
	maxRequestToolCallsPerMsg    = 16384
	maxRequestToolCallsTotal     = 65536
	maxRequestTools              = 4096
	maxRequestStopSequences      = 4096
	maxResponseCandidates        = 256
	maxResponseParts             = 16384
	maxResponsePartFieldsPerPart = 1024
	maxResponsePartFieldsTotal   = 65536
)

type jsonArrayLimit struct {
	segments    []string
	perArray    int
	totalMax    int
	description string
}

type jsonObjectLimit struct {
	segments    []string
	perObject   int
	totalMax    int
	description string
}

var requestArrayLimits = []jsonArrayLimit{
	{segments: []string{"messages"}, perArray: maxRequestMessages, totalMax: maxRequestMessages, description: "request messages"},
	{segments: []string{"messages", "*", "content"}, perArray: maxRequestContentPartsPerMsg, totalMax: maxRequestContentPartsTotal, description: "request content parts"},
	{segments: []string{"messages", "*", "tool_calls"}, perArray: maxRequestToolCallsPerMsg, totalMax: maxRequestToolCallsTotal, description: "request tool calls"},
	{segments: []string{"tools"}, perArray: maxRequestTools, totalMax: maxRequestTools, description: "request tools"},
	{segments: []string{"stop"}, perArray: maxRequestStopSequences, totalMax: maxRequestStopSequences, description: "request stop sequences"},
}

var responseArrayLimits = []jsonArrayLimit{
	{segments: []string{"candidates"}, perArray: maxResponseCandidates, totalMax: maxResponseCandidates, description: "response candidates"},
	{segments: []string{"candidates", "*", "content", "parts"}, perArray: maxResponseParts, totalMax: maxResponseParts, description: "response parts"},
}

var requestObjectLimits = []jsonObjectLimit{
	// The root object is decoded into map[string]json.RawMessage before the
	// typed request is built. Bound its field count during the structural scan
	// so unknown keys cannot force one map entry (and value copy) per field.
	{perObject: maxRequestTopLevelFields, totalMax: maxRequestTopLevelFields, description: "request top-level fields"},
	// generationConfig is decoded into its own map[string]json.RawMessage.
	// Give it the same order of magnitude as the root budget so a small root
	// with a wide nested object cannot amplify memory far beyond the raw
	// request size. Repeated keys count too, matching the root scanner.
	{segments: []string{"generationConfig"}, perObject: maxRequestGenerationConfig, totalMax: maxRequestGenerationConfig, description: "request generationConfig fields"},
}

var responseObjectLimits = []jsonObjectLimit{
	{segments: []string{"candidates", "*", "content", "parts", "*"}, perObject: maxResponsePartFieldsPerPart, totalMax: maxResponsePartFieldsTotal, description: "response part fields"},
}

type jsonBudgetScanner struct {
	data  []byte
	pos   int
	path  []string
	depth int

	arrayLimits  []jsonArrayLimit
	arrayTotals  []int
	objectLimits []jsonObjectLimit
	objectTotals []int
}

func checkChatRequestBudgets(data []byte) error {
	return scanJSONBudgets(data, requestArrayLimits, requestObjectLimits)
}

func checkGeminiResponseBudgets(data []byte) error {
	return scanJSONBudgets(data, responseArrayLimits, responseObjectLimits)
}

func scanJSONBudgets(data []byte, arrayLimits []jsonArrayLimit, objectLimits []jsonObjectLimit) error {
	s := &jsonBudgetScanner{
		data:         data,
		arrayLimits:  arrayLimits,
		arrayTotals:  make([]int, len(arrayLimits)),
		objectLimits: objectLimits,
		objectTotals: make([]int, len(objectLimits)),
	}
	if err := s.parseValue(); err != nil {
		return err
	}
	s.skipSpace()
	if s.pos != len(s.data) {
		return fmt.Errorf("unexpected data after JSON value")
	}
	return nil
}

func (s *jsonBudgetScanner) skipSpace() {
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case ' ', '\t', '\n', '\r':
			s.pos++
		default:
			return
		}
	}
}

func (s *jsonBudgetScanner) enterNesting() error {
	s.depth++
	if s.depth > maxJSONNestingDepth {
		s.depth--
		return fmt.Errorf("JSON nesting depth exceeds %d", maxJSONNestingDepth)
	}
	return nil
}

func (s *jsonBudgetScanner) leaveNesting() {
	s.depth--
}

func (s *jsonBudgetScanner) parseValue() error {
	s.skipSpace()
	if s.pos >= len(s.data) {
		return fmt.Errorf("unexpected end of JSON input")
	}
	switch s.data[s.pos] {
	case '{':
		return s.parseObject()
	case '[':
		return s.parseArray()
	case '"':
		_, err := s.scanString()
		return err
	case 't':
		return s.scanLiteral("true")
	case 'f':
		return s.scanLiteral("false")
	case 'n':
		return s.scanLiteral("null")
	default:
		return s.scanNumber()
	}
}

func (s *jsonBudgetScanner) parseObject() error {
	if err := s.enterNesting(); err != nil {
		return err
	}
	defer s.leaveNesting()

	s.pos++
	limitIndex, limited := s.matchObjectLimit()
	var fields int
	s.skipSpace()
	if s.pos < len(s.data) && s.data[s.pos] == '}' {
		s.pos++
		return nil
	}
	for {
		key, err := s.parseKey()
		if err != nil {
			return err
		}
		s.skipSpace()
		if s.pos >= len(s.data) || s.data[s.pos] != ':' {
			return fmt.Errorf("expected ':'")
		}
		s.pos++

		if limited {
			limit := s.objectLimits[limitIndex]
			fields++
			if fields > limit.perObject {
				return fmt.Errorf("%s exceed limit of %d per object", limit.description, limit.perObject)
			}
			s.objectTotals[limitIndex]++
			if s.objectTotals[limitIndex] > limit.totalMax {
				return fmt.Errorf("%s exceed total limit of %d", limit.description, limit.totalMax)
			}
		}

		s.path = append(s.path, key)
		err = s.parseValue()
		s.path = s.path[:len(s.path)-1]
		if err != nil {
			return err
		}

		s.skipSpace()
		if s.pos >= len(s.data) {
			return fmt.Errorf("unexpected end of JSON input")
		}
		switch s.data[s.pos] {
		case ',':
			s.pos++
			s.skipSpace()
		case '}':
			s.pos++
			return nil
		default:
			return fmt.Errorf("expected ',' or '}'")
		}
	}
}

func (s *jsonBudgetScanner) parseArray() error {
	if err := s.enterNesting(); err != nil {
		return err
	}
	defer s.leaveNesting()

	s.pos++
	limitIndex, limited := s.matchArrayLimit()
	var count int
	s.skipSpace()
	if s.pos < len(s.data) && s.data[s.pos] == ']' {
		s.pos++
		return nil
	}
	for {
		count++
		if limited {
			limit := s.arrayLimits[limitIndex]
			if count > limit.perArray {
				return fmt.Errorf("%s exceed limit of %d", limit.description, limit.perArray)
			}
			s.arrayTotals[limitIndex]++
			if s.arrayTotals[limitIndex] > limit.totalMax {
				return fmt.Errorf("%s exceed total limit of %d", limit.description, limit.totalMax)
			}
		}

		s.path = append(s.path, "*")
		err := s.parseValue()
		s.path = s.path[:len(s.path)-1]
		if err != nil {
			return err
		}

		s.skipSpace()
		if s.pos >= len(s.data) {
			return fmt.Errorf("unexpected end of JSON input")
		}
		switch s.data[s.pos] {
		case ',':
			s.pos++
			s.skipSpace()
		case ']':
			s.pos++
			return nil
		default:
			return fmt.Errorf("expected ',' or ']'")
		}
	}
}

func (s *jsonBudgetScanner) parseKey() (string, error) {
	start := s.pos
	end, err := s.scanString()
	if err != nil {
		return "", err
	}
	raw := s.data[start:end]
	// Field names that matter here are short. Avoid decoding arbitrarily large
	// payload keys; such a key cannot match a budgeted field name anyway.
	if len(raw) > 128 {
		return "", nil
	}
	var key string
	if err := json.Unmarshal(raw, &key); err != nil {
		return "", err
	}
	return foldJSONFieldName(key), nil
}

// foldJSONFieldName mirrors the field-name folding used by encoding/json when
// it matches object keys against struct fields. Lowercasing alone is not
// enough: Unicode simple case folding makes keys such as "meſſages" decode as
// "messages", so the budget scanner must use the same equivalence to avoid
// unscanned spellings bypassing the decoded-size limits.
func foldJSONFieldName(name string) string {
	var b strings.Builder
	b.Grow(len(name))
	for _, r := range name {
		if r < utf8.RuneSelf {
			if 'a' <= r && r <= 'z' {
				r -= 'a' - 'A'
			}
			b.WriteRune(r)
			continue
		}
		b.WriteRune(foldJSONRune(r))
	}
	return b.String()
}

func foldJSONRune(r rune) rune {
	for {
		r2 := unicode.SimpleFold(r)
		if r2 <= r {
			return r2
		}
		r = r2
	}
}

func (s *jsonBudgetScanner) scanString() (int, error) {
	if s.pos >= len(s.data) || s.data[s.pos] != '"' {
		return 0, fmt.Errorf("expected string")
	}
	s.pos++
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case '"':
			s.pos++
			return s.pos, nil
		case '\\':
			s.pos++
			if s.pos >= len(s.data) {
				return 0, fmt.Errorf("unexpected end of JSON input")
			}
			s.pos++
		default:
			s.pos++
		}
	}
	return 0, fmt.Errorf("unexpected end of JSON input")
}

func (s *jsonBudgetScanner) scanLiteral(literal string) error {
	if len(s.data)-s.pos < len(literal) || string(s.data[s.pos:s.pos+len(literal)]) != literal {
		return fmt.Errorf("invalid JSON value")
	}
	s.pos += len(literal)
	return nil
}

func (s *jsonBudgetScanner) scanNumber() error {
	start := s.pos
scan:
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case ' ', '\t', '\n', '\r', ',', '}', ']':
			break scan
		default:
			s.pos++
		}
	}
	if s.pos == start {
		return fmt.Errorf("invalid JSON value")
	}
	return nil
}

// jsonObjectHasTopLevelKey reports whether data is a JSON object with key at
// its top level. It scans the object structurally and skips values without
// constructing maps or slices, so a wide business object does not allocate one
// Go value per field just to answer a type check.
func jsonObjectHasTopLevelKey(data []byte, key string) bool {
	s := &jsonBudgetScanner{data: data}
	s.skipSpace()
	if s.pos >= len(s.data) || s.data[s.pos] != '{' {
		return false
	}
	s.pos++
	s.skipSpace()
	if s.pos < len(s.data) && s.data[s.pos] == '}' {
		return false
	}
	var found bool
	for {
		start := s.pos
		if _, err := s.scanString(); err != nil {
			return false
		}
		if rawJSONStringEquals(s.data[start:s.pos], key) {
			found = true
		}
		s.skipSpace()
		if s.pos >= len(s.data) || s.data[s.pos] != ':' {
			return false
		}
		s.pos++
		if err := s.skipValue(); err != nil {
			return false
		}
		s.skipSpace()
		if s.pos >= len(s.data) {
			return false
		}
		switch s.data[s.pos] {
		case ',':
			s.pos++
			s.skipSpace()
		case '}':
			s.pos++
			s.skipSpace()
			return found && s.pos == len(s.data)
		default:
			return false
		}
	}
}

func rawJSONStringEquals(raw []byte, want string) bool {
	if len(raw) < 2 || raw[0] != '"' || raw[len(raw)-1] != '"' {
		return false
	}
	body := raw[1 : len(raw)-1]
	escaped := false
	for _, c := range body {
		if c == '\\' {
			escaped = true
			break
		}
	}
	if !escaped {
		if len(body) != len(want) {
			return false
		}
		for i := range body {
			if body[i] != want[i] {
				return false
			}
		}
		return true
	}
	var decoded string
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return false
	}
	return decoded == want
}

func (s *jsonBudgetScanner) skipValue() error {
	s.skipSpace()
	if s.pos >= len(s.data) {
		return fmt.Errorf("unexpected end of JSON input")
	}
	switch s.data[s.pos] {
	case '{':
		return s.skipObject()
	case '[':
		return s.skipArray()
	case '"':
		_, err := s.scanString()
		return err
	case 't':
		return s.scanLiteral("true")
	case 'f':
		return s.scanLiteral("false")
	case 'n':
		return s.scanLiteral("null")
	default:
		return s.scanNumber()
	}
}

func (s *jsonBudgetScanner) skipObject() error {
	if err := s.enterNesting(); err != nil {
		return err
	}
	defer s.leaveNesting()

	s.pos++
	s.skipSpace()
	if s.pos < len(s.data) && s.data[s.pos] == '}' {
		s.pos++
		return nil
	}
	for {
		if _, err := s.scanString(); err != nil {
			return err
		}
		s.skipSpace()
		if s.pos >= len(s.data) || s.data[s.pos] != ':' {
			return fmt.Errorf("expected ':'")
		}
		s.pos++
		if err := s.skipValue(); err != nil {
			return err
		}
		s.skipSpace()
		if s.pos >= len(s.data) {
			return fmt.Errorf("unexpected end of JSON input")
		}
		switch s.data[s.pos] {
		case ',':
			s.pos++
			s.skipSpace()
		case '}':
			s.pos++
			return nil
		default:
			return fmt.Errorf("expected ',' or '}'")
		}
	}
}

func (s *jsonBudgetScanner) skipArray() error {
	if err := s.enterNesting(); err != nil {
		return err
	}
	defer s.leaveNesting()

	s.pos++
	s.skipSpace()
	if s.pos < len(s.data) && s.data[s.pos] == ']' {
		s.pos++
		return nil
	}
	for {
		if err := s.skipValue(); err != nil {
			return err
		}
		s.skipSpace()
		if s.pos >= len(s.data) {
			return fmt.Errorf("unexpected end of JSON input")
		}
		switch s.data[s.pos] {
		case ',':
			s.pos++
			s.skipSpace()
		case ']':
			s.pos++
			return nil
		default:
			return fmt.Errorf("expected ',' or ']'")
		}
	}
}

func (s *jsonBudgetScanner) matchArrayLimit() (int, bool) {
	for i := range s.arrayLimits {
		if pathMatches(s.path, s.arrayLimits[i].segments) {
			return i, true
		}
	}
	return 0, false
}

func (s *jsonBudgetScanner) matchObjectLimit() (int, bool) {
	for i := range s.objectLimits {
		if pathMatches(s.path, s.objectLimits[i].segments) {
			return i, true
		}
	}
	return 0, false
}

func pathMatches(path, segments []string) bool {
	if len(path) != len(segments) {
		return false
	}
	for i := range path {
		if path[i] != foldJSONFieldName(segments[i]) {
			return false
		}
	}
	return true
}

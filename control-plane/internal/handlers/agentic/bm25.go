package agentic

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// BM25 saturation parameters. k1 controls term-frequency saturation and b the
// document-length normalization — the standard defaults work well for the small
// reasoner corpus (a few thousand docs at most), which is rebuilt per request.
const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// searchField is one weighted text field of a searchable document. The boost is
// applied at term-frequency time, giving a BM25F-lite ranker: a term found in a
// high-boost field (e.g. the reasoner id) contributes proportionally more than
// the same term in a low-boost field (e.g. the owning agent id).
type searchField struct {
	boost float64
	text  string
}

// searchDoc is one document fed to the index, identified by an opaque stable id
// (used both for tie-breaking and for mapping a hit back to its source record).
type searchDoc struct {
	id     string
	fields []searchField
}

// searchHit is a scored document, highest score first.
type searchHit struct {
	id    string
	score float64
}

// scoredDoc is the indexed form of a searchDoc: per-term weighted frequencies
// plus the (boost-weighted) document length used for length normalization.
type scoredDoc struct {
	id      string
	weights map[string]float64
	length  float64
}

// bm25Index is a self-contained, in-memory BM25F-lite index. It carries no
// external dependencies and is cheap enough to rebuild on every request.
type bm25Index struct {
	k1     float64
	b      float64
	docs   []scoredDoc
	df     map[string]int
	numDoc int
	avgdl  float64
}

// newBM25Index tokenizes and indexes the supplied documents. Field boosts are
// folded into each term's weighted frequency, and document frequency counts a
// term once per document regardless of how many fields it appears in.
func newBM25Index(docs []searchDoc) *bm25Index {
	idx := &bm25Index{
		k1:     bm25K1,
		b:      bm25B,
		df:     make(map[string]int),
		numDoc: len(docs),
	}

	var totalLen float64
	for _, d := range docs {
		weights := make(map[string]float64)
		var length float64
		for _, f := range d.fields {
			toks := tokenize(f.text)
			if len(toks) == 0 {
				continue
			}
			length += float64(len(toks)) * f.boost
			for _, t := range toks {
				weights[t] += f.boost
			}
		}
		for t := range weights {
			idx.df[t]++
		}
		idx.docs = append(idx.docs, scoredDoc{id: d.id, weights: weights, length: length})
		totalLen += length
	}

	if idx.numDoc > 0 {
		idx.avgdl = totalLen / float64(idx.numDoc)
	}
	return idx
}

// Search ranks every indexed document against the free-text query and returns
// the matches (score > 0) ordered by descending score, ties broken by ascending
// document id for deterministic output.
func (idx *bm25Index) Search(query string) []searchHit {
	if idx.numDoc == 0 {
		return nil
	}
	qterms := uniqueTokens(query)
	if len(qterms) == 0 {
		return nil
	}

	hits := make([]searchHit, 0)
	for _, d := range idx.docs {
		var score float64
		for _, qt := range qterms {
			wtf, ok := d.weights[qt]
			if !ok || wtf <= 0 {
				continue
			}
			df := idx.df[qt]
			idf := math.Log(1 + (float64(idx.numDoc)-float64(df)+0.5)/(float64(df)+0.5))

			lengthRatio := 1.0
			if idx.avgdl > 0 {
				lengthRatio = d.length / idx.avgdl
			}
			denom := wtf + idx.k1*(1-idx.b+idx.b*lengthRatio)
			if denom == 0 {
				continue
			}
			score += idf * (wtf * (idx.k1 + 1)) / denom
		}
		if score > 0 {
			hits = append(hits, searchHit{id: d.id, score: score})
		}
	}

	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].id < hits[j].id
	})
	return hits
}

// tokenize lowercases and splits text on non-alphanumeric boundaries, then
// further splits camelCase / PascalCase / acronym runs. This makes reasoner ids
// like "run_pr_resolver" and "reviewPRRequest" match natural-language queries
// such as "pr resolve" or "review request".
func tokenize(s string) []string {
	if s == "" {
		return nil
	}
	fields := strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	tokens := make([]string, 0, len(fields))
	for _, f := range fields {
		for _, part := range splitCamel(f) {
			part = strings.ToLower(part)
			if part != "" {
				tokens = append(tokens, part)
			}
		}
	}
	return tokens
}

// uniqueTokens tokenizes and de-duplicates, preserving first-seen order. Query
// terms are scored once each regardless of repetition.
func uniqueTokens(s string) []string {
	toks := tokenize(s)
	seen := make(map[string]struct{}, len(toks))
	out := make([]string, 0, len(toks))
	for _, t := range toks {
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		out = append(out, t)
	}
	return out
}

// splitCamel breaks a single alphanumeric word at camelCase and acronym
// boundaries: "reviewPR" -> [review, PR]; "PRReview" -> [PR, Review];
// "v2Handler" -> [v2, Handler]. Words with no internal boundary pass through
// unchanged.
func splitCamel(word string) []string {
	if word == "" {
		return nil
	}
	runes := []rune(word)
	if len(runes) == 1 {
		return []string{word}
	}

	var parts []string
	start := 0
	for i := 1; i < len(runes); i++ {
		prev := runes[i-1]
		cur := runes[i]
		boundary := false
		switch {
		case (unicode.IsLower(prev) || unicode.IsDigit(prev)) && unicode.IsUpper(cur):
			// lowercase/digit -> uppercase: "foo|Bar", "v2|Handler"
			boundary = true
		case unicode.IsUpper(prev) && unicode.IsUpper(cur) && i+1 < len(runes) && unicode.IsLower(runes[i+1]):
			// acronym -> word: "HTTP|Server"
			boundary = true
		}
		if boundary {
			parts = append(parts, string(runes[start:i]))
			start = i
		}
	}
	parts = append(parts, string(runes[start:]))
	return parts
}

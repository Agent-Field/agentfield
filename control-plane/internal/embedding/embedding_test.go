package embedding

import (
	"context"
	"math"
	"testing"
)

func TestFakeEmbedderDimensionAndDeterminism(t *testing.T) {
	e := NewFakeEmbedder()
	if e.Dimensions() != Dimensions {
		t.Fatalf("Dimensions() = %d, want %d", e.Dimensions(), Dimensions)
	}

	ctx := context.Background()
	got, err := e.Embed(ctx, []string{"hello world", "hello world", "different text"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d vectors, want 3", len(got))
	}
	for i, v := range got {
		if len(v) != Dimensions {
			t.Fatalf("vector %d dim = %d, want %d", i, len(v), Dimensions)
		}
	}

	// Determinism: identical text -> identical vector.
	for i := range got[0] {
		if got[0][i] != got[1][i] {
			t.Fatalf("same text produced different vectors at index %d", i)
		}
	}

	// Different text -> different vector (overwhelmingly likely).
	same := true
	for i := range got[0] {
		if got[0][i] != got[2][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different text produced identical vector")
	}

	// Vectors are L2-normalized (unit length within float tolerance).
	var norm float64
	for _, x := range got[0] {
		norm += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(norm)-1.0) > 1e-3 {
		t.Fatalf("vector not unit-normalized: norm=%v", math.Sqrt(norm))
	}
}

func TestNewFromConfigFallback(t *testing.T) {
	// No key, auto -> fake.
	emb, isOpenAI := NewFromConfig(ProviderConfig{})
	if isOpenAI {
		t.Fatal("expected fake embedder when no API key configured")
	}
	if _, ok := emb.(*FakeEmbedder); !ok {
		t.Fatalf("expected *FakeEmbedder, got %T", emb)
	}

	// provider=openai but no key -> still fake.
	emb, isOpenAI = NewFromConfig(ProviderConfig{Provider: "openai"})
	if isOpenAI {
		t.Fatal("expected fallback to fake when provider=openai but no key")
	}
	if _, ok := emb.(*FakeEmbedder); !ok {
		t.Fatalf("expected *FakeEmbedder, got %T", emb)
	}

	// key present -> openai.
	emb, isOpenAI = NewFromConfig(ProviderConfig{APIKey: "sk-test"})
	if !isOpenAI {
		t.Fatal("expected openai embedder when key present")
	}
	if emb.Dimensions() != Dimensions {
		t.Fatalf("openai dim = %d, want %d", emb.Dimensions(), Dimensions)
	}

	// provider=fake forces fake even with key.
	_, isOpenAI = NewFromConfig(ProviderConfig{Provider: "fake", APIKey: "sk-test"})
	if isOpenAI {
		t.Fatal("expected fake when provider=fake despite key")
	}
}

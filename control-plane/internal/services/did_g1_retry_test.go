package services

import (
	"context"
	"database/sql"
	"fmt"
	"hash/fnv"
	"net/url"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/stretchr/testify/require"
)

// Delay real SQLite writes so queued registrations outlive the test deadline.
type slowG1DIDStorage struct {
	storage.StorageProvider
}

func (s *slowG1DIDStorage) UpsertAgentDIDWithComponents(ctx context.Context, record storage.AgentDIDRecord) error {
	timer := time.NewTimer(10 * time.Millisecond)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
	}
	return s.StorageProvider.UpsertAgentDIDWithComponents(ctx, record)
}

func TestG1DIDQueuedRegistrationsDoNotTimeOut(t *testing.T) {
	originalTimeout := didRegistrationTimeout
	didRegistrationTimeout = 200 * time.Millisecond
	t.Cleanup(func() { didRegistrationTimeout = originalTimeout })

	for _, partial := range []bool{false, true} {
		name := "full"
		if partial {
			name = "partial"
		}
		t.Run(name, func(t *testing.T) {
			provider := openReproStorage(t, t.TempDir())
			t.Cleanup(func() { require.NoError(t, provider.Close(context.Background())) })
			svc, _ := openReproService(t, provider, "server-a", filepath.Join(t.TempDir(), "keys"))
			const count = 100
			if partial {
				for i := 0; i < count; i++ {
					register(t, svc, fmt.Sprintf("node-%d", i), nil, nil)
				}
			}
			svc.registry.storageProvider = &slowG1DIDStorage{StorageProvider: provider}

			start := make(chan struct{})
			responses := make([]*types.DIDRegistrationResponse, count)
			failures := make([]error, count)
			elapsed := make([]time.Duration, count)
			var wg sync.WaitGroup
			for i := 0; i < count; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					<-start
					began := time.Now()
					nodeID := fmt.Sprintf("node-%d", i)
					if partial {
						responses[i], failures[i] = svc.PartialRegisterAgent(&types.PartialDIDRegistrationRequest{
							AgentNodeID: nodeID, NewReasonerIDs: []string{"r"}, AllReasoners: []types.ReasonerDefinition{{ID: "r"}},
						})
					} else {
						responses[i], failures[i] = svc.RegisterAgent(&types.DIDRegistrationRequest{
							AgentNodeID: nodeID, Reasoners: []types.ReasonerDefinition{{ID: "r"}},
						})
					}
					elapsed[i] = time.Since(began)
				}(i)
			}
			close(start)
			wg.Wait()
			var longest time.Duration
			for i, response := range responses {
				require.NoError(t, failures[i])
				require.NotNil(t, response)
				require.True(t, response.Success, "registration %d: %s", i, response.Error)
				longest = max(longest, elapsed[i])
			}
			require.Greater(t, longest, 5*didRegistrationTimeout, "the queue must outlast the registration deadline")
			distinctDIDRows(t, provider, count, count)
			t.Logf("%d/%d registrations succeeded; longest queued registration: %s (deadline %s)", count, count, longest, didRegistrationTimeout)
		})
	}
}

func TestG1DIDSerializablePostgres(t *testing.T) {
	base, _, dsn := postgresTakeoverStores(t)
	keyDir := filepath.Join(t.TempDir(), "keys")
	openReproService(t, base, "server-a", keyDir)
	u, err := url.Parse(dsn)
	require.NoError(t, err)
	query := u.Query()
	query.Set("options", "-c default_transaction_isolation=serializable")
	u.RawQuery = query.Encode()
	vectorOff := false
	cfg := storage.StorageConfig{Mode: "postgres", Postgres: storage.PostgresStorageConfig{DSN: u.String()}, Vector: storage.VectorStoreConfig{Enabled: &vectorOff}}
	provider := storage.NewPostgresStorage(cfg.Postgres)
	require.NoError(t, provider.Initialize(context.Background(), cfg))
	t.Cleanup(func() { require.NoError(t, provider.Close(context.Background())) })
	db, err := sql.Open("pgx", u.String())
	require.NoError(t, err)
	defer db.Close()
	var isolation string
	require.NoError(t, db.QueryRow("SHOW default_transaction_isolation").Scan(&isolation))
	require.Equal(t, "serializable", isolation)

	// Separate registries prevent an in-memory mutex from serializing replicas.
	const count = 40
	services := make([]*DIDService, count)
	for i := range services {
		services[i], _ = openReproService(t, provider, "server-a", keyDir)
	}
	start := make(chan struct{})
	responses := make([]*types.DIDRegistrationResponse, count)
	errors := make([]error, count)
	elapsed := make([]time.Duration, count)
	var wg sync.WaitGroup
	for i, svc := range services {
		wg.Add(1)
		go func(i int, svc *DIDService) {
			defer wg.Done()
			<-start
			began := time.Now()
			responses[i], errors[i] = svc.RegisterAgent(&types.DIDRegistrationRequest{
				AgentNodeID: fmt.Sprintf("node-%d", i), Reasoners: []types.ReasonerDefinition{{ID: "r"}},
			})
			elapsed[i] = time.Since(began)
		}(i, svc)
	}
	close(start)
	wg.Wait()
	for i, resp := range responses {
		require.NoError(t, errors[i])
		require.NotNil(t, resp)
		require.True(t, resp.Success, "registration %d: %s", i, resp.Error)
		require.Less(t, elapsed[i], didRegistrationTimeout+time.Second)
	}
	distinctDIDRows(t, provider, count, count)
}

func TestG1DIDRetryPolicies(t *testing.T) {
	for _, code := range []string{"40001", "40P01", "55P03"} {
		t.Run(code, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			calls := 0
			response, err := retryDIDAllocation(ctx, func() (*types.DIDRegistrationResponse, error) {
				calls++
				if calls < 5 {
					return nil, fmt.Errorf("transaction: %w", &pgconn.PgError{Code: code, Message: "last database error"})
				}
				return &types.DIDRegistrationResponse{Success: true}, nil
			})
			require.NoError(t, err)
			if code == "55P03" {
				require.False(t, response.Success)
				require.Equal(t, 2, calls)
				require.Contains(t, response.Error, "last database error")
			} else {
				require.True(t, response.Success, response.Error)
				require.Equal(t, 5, calls)
			}
		})
	}
}

func TestG1DIDRegistrationDeadlinePostgres(t *testing.T) {
	provider, _, dsn := postgresTakeoverStores(t)
	svc, _ := openReproService(t, provider, "server-a", filepath.Join(t.TempDir(), "keys"))
	db, err := sql.Open("pgx", dsn)
	require.NoError(t, err)
	defer db.Close()
	tx, err := db.Begin()
	require.NoError(t, err)
	defer tx.Rollback()
	hash := fnv.New64a()
	_, _ = hash.Write([]byte("agentfield:did-server:server-a"))
	_, err = tx.Exec("SELECT pg_advisory_xact_lock($1)", int64(hash.Sum64()))
	require.NoError(t, err)
	began := time.Now()
	response, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "blocked"})
	require.NoError(t, err)
	require.False(t, response.Success)
	require.Contains(t, response.Error, "55P03", "retain the database error from the first lock timeout")
	require.Contains(t, response.Error, "deadline")
	require.GreaterOrEqual(t, time.Since(began), didRegistrationTimeout-time.Second)
	require.Less(t, time.Since(began), didRegistrationTimeout+3*time.Second)
	require.NoError(t, tx.Rollback())
	register(t, svc, "blocked", nil, nil)
}

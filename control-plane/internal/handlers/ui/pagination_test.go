package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestParsePage(t *testing.T) {
	tests := []struct {
		value string
		want  int
	}{
		{"", 1},
		{"abc", 1},
		{"1.5", 1},
		{"0", 1},
		{"-3", 1},
		{"1", 1},
		{" 7 ", 7},
		{"1000000", maxPage},
		{"1000001", maxPage},
		{"99999999999999999999", maxPage},
		{"-99999999999999999999", 1},
	}
	for _, tt := range tests {
		require.Equal(t, tt.want, parsePage(tt.value), "parsePage(%q)", tt.value)
	}
}

// pagingProbeStorage records the offset or page each paginated handler hands to storage.
type pagingProbeStorage struct {
	storage.StorageProvider
	offset      int
	historyPage int
}

func (s *pagingProbeStorage) QueryRunSummaries(_ context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	s.offset = filter.Offset
	return nil, 5, nil
}

func (s *pagingProbeStorage) QueryExecutionRecords(_ context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	s.offset = filter.Offset
	return nil, nil
}

func (s *pagingProbeStorage) GetReasonerExecutionHistory(_ context.Context, _ string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	s.historyPage = page
	s.offset = (page - 1) * limit // mirrors LocalStorage
	return &types.ReasonerExecutionHistory{Page: page, Limit: limit}, nil
}

// Every endpoint that turns ?page= into an offset must bound it. An unbounded
// page can overflow (page-1)*size into a negative offset, and one too large for
// int must not fall back to the first page.
func TestPaginatedHandlersBoundPage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoints := []struct {
		name    string
		route   string
		url     string // ends ready for "&page=..."
		size    int
		handler func(storage.StorageProvider) gin.HandlerFunc
	}{
		{
			"workflow runs", "/runs", "/runs?page_size=200", 200,
			func(s storage.StorageProvider) gin.HandlerFunc {
				return NewWorkflowRunHandler(s).ListWorkflowRunsHandler
			},
		},
		{
			"agent executions", "/agents/:agentId/executions", "/agents/a/executions?pageSize=100", 100,
			func(s storage.StorageProvider) gin.HandlerFunc {
				return NewExecutionHandler(s, nil, nil).ListExecutionsHandler
			},
		},
		{
			"executions summary", "/summary", "/summary?page_size=100", 100,
			func(s storage.StorageProvider) gin.HandlerFunc {
				return NewExecutionHandler(s, nil, nil).GetExecutionsSummaryHandler
			},
		},
		{
			"enhanced executions", "/enhanced", "/enhanced?limit=200", 200,
			func(s storage.StorageProvider) gin.HandlerFunc {
				return NewExecutionHandler(s, nil, nil).GetEnhancedExecutionsHandler
			},
		},
		{
			"reasoner history", "/reasoners/:reasonerId/history", "/reasoners/node.r/history?limit=100", 100,
			func(s storage.StorageProvider) gin.HandlerFunc {
				return NewReasonersHandler(s).GetExecutionHistoryHandler
			},
		},
	}

	pages := []struct {
		name     string
		value    string
		wantPage int
	}{
		{"ordinary", "3", 3},
		{"past the cap", "1000001", 1_000_000},
		{"overflows the offset", "9223372036854775807", 1_000_000},
		{"too large for int", "99999999999999999999", 1_000_000},
		{"too small for int", "-99999999999999999999", 1},
	}

	for _, ep := range endpoints {
		for _, p := range pages {
			t.Run(ep.name+"/"+p.name, func(t *testing.T) {
				store := &pagingProbeStorage{StorageProvider: setupTestStorage(t)}
				router := gin.New()
				router.GET(ep.route, ep.handler(store))

				rec := httptest.NewRecorder()
				router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, ep.url+"&page="+p.value, nil))
				require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

				var body struct {
					Page    int  `json:"page"`
					HasMore bool `json:"has_more"`
					HasNext bool `json:"has_next"`
				}
				require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
				require.Equal(t, (p.wantPage-1)*ep.size, store.offset, "offset sent to storage")
				require.Equal(t, p.wantPage, body.Page, "page reported to the client")
				require.False(t, body.HasMore || body.HasNext, "nothing follows an empty result")
			})
		}
	}
}

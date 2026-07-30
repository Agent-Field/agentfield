package ui

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"github.com/gin-gonic/gin"
)

type stubPackageJobManager struct {
	startInstall func(string, bool) (*packagejobs.Job, error)
	startUpdate  func(string) (*packagejobs.Job, error)
	uninstall    func(string) error
	jobs         map[string]*packagejobs.Job
}

func (s *stubPackageJobManager) StartInstall(source string, force bool) (*packagejobs.Job, error) {
	return s.startInstall(source, force)
}
func (s *stubPackageJobManager) StartUpdate(name string) (*packagejobs.Job, error) {
	return s.startUpdate(name)
}
func (s *stubPackageJobManager) Uninstall(name string) error { return s.uninstall(name) }
func (s *stubPackageJobManager) GetJob(id string) (*packagejobs.Job, bool) {
	job, ok := s.jobs[id]
	return job, ok
}
func (s *stubPackageJobManager) ListJobs() []*packagejobs.Job {
	result := make([]*packagejobs.Job, 0, len(s.jobs))
	for _, job := range s.jobs {
		result = append(result, job)
	}
	return result
}

func testContext(method, path string, body []byte, params ...gin.Param) (*gin.Context, *httptest.ResponseRecorder) {
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(method, path, bytes.NewReader(body))
	ctx.Request.Header.Set("Content-Type", "application/json")
	ctx.Params = params
	return ctx, recorder
}

// Contract 1: POST install returns 202 and a job ID.
func TestInstallPackageHandlerAccepted(t *testing.T) {
	manager := &stubPackageJobManager{
		startInstall: func(source string, force bool) (*packagejobs.Job, error) {
			if source != "https://github.com/owner/repo" || force {
				t.Fatalf("request source=%q force=%v", source, force)
			}
			return &packagejobs.Job{ID: "job-1"}, nil
		},
	}
	ctx, response := testContext(http.MethodPost, "/", []byte(`{"source":"https://github.com/owner/repo"}`))
	NewPackageInstallHandler(manager).InstallPackageHandler(ctx)
	if response.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var body map[string]string
	_ = json.Unmarshal(response.Body.Bytes(), &body)
	if body["job_id"] != "job-1" {
		t.Fatalf("body=%v", body)
	}
}

// Contracts 2 and 3: validation maps to 400 and busy maps to 409.
func TestInstallPackageHandlerErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want int
	}{
		{"invalid", packagejobs.ErrInvalidSource, http.StatusBadRequest},
		{"busy", packagejobs.ErrBusy, http.StatusConflict},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manager := &stubPackageJobManager{
				startInstall: func(string, bool) (*packagejobs.Job, error) { return nil, test.err },
			}
			ctx, response := testContext(http.MethodPost, "/", []byte(`{"source":"bad"}`))
			NewPackageInstallHandler(manager).InstallPackageHandler(ctx)
			if response.Code != test.want {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

// Contract 5: GET of an unknown job returns 404.
func TestGetInstallJobHandlerUnknown(t *testing.T) {
	manager := &stubPackageJobManager{jobs: map[string]*packagejobs.Job{}}
	ctx, response := testContext(http.MethodGet, "/", nil, gin.Param{Key: "jobId", Value: "missing"})
	NewPackageInstallHandler(manager).GetInstallJobHandler(ctx)
	if response.Code != http.StatusNotFound {
		t.Fatalf("status=%d", response.Code)
	}
}

// Contract 7: unknown uninstall targets map to 404.
func TestUninstallPackageHandlerUnknown(t *testing.T) {
	manager := &stubPackageJobManager{uninstall: func(string) error { return packagejobs.ErrNotFound }}
	ctx, response := testContext(http.MethodDelete, "/", nil, gin.Param{Key: "packageId", Value: "missing"})
	NewPackageInstallHandler(manager).UninstallPackageHandler(ctx)
	if response.Code != http.StatusNotFound {
		t.Fatalf("status=%d", response.Code)
	}
}

func TestUpdatePackageHandlerMappings(t *testing.T) {
	tests := []struct {
		name string
		job  *packagejobs.Job
		err  error
		want int
	}{
		{"accepted", &packagejobs.Job{ID: "update-1"}, nil, http.StatusAccepted},
		{"missing", nil, packagejobs.ErrNotFound, http.StatusNotFound},
		{"busy", nil, packagejobs.ErrBusy, http.StatusConflict},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manager := &stubPackageJobManager{
				startUpdate: func(string) (*packagejobs.Job, error) { return test.job, test.err },
				uninstall:   func(string) error { return errors.New("unused") },
			}
			ctx, response := testContext(http.MethodPost, "/", nil, gin.Param{Key: "packageId", Value: "demo"})
			NewPackageInstallHandler(manager).UpdatePackageHandler(ctx)
			if response.Code != test.want {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

// Exercises the malformed-JSON bind-error branch before the manager is called.
func TestInstallPackageHandlerRejectsMalformedJSON(t *testing.T) {
	manager := &stubPackageJobManager{
		startInstall: func(string, bool) (*packagejobs.Job, error) {
			t.Fatal("manager should not be called")
			return nil, nil
		},
	}
	ctx, response := testContext(http.MethodPost, "/", []byte(`{`))
	NewPackageInstallHandler(manager).InstallPackageHandler(ctx)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

// Exercises the successful get and list handler branches.
func TestPackageInstallJobReadHandlers(t *testing.T) {
	job := &packagejobs.Job{ID: "job-1"}
	manager := &stubPackageJobManager{jobs: map[string]*packagejobs.Job{"job-1": job}}
	handler := NewPackageInstallHandler(manager)

	ctx, response := testContext(http.MethodGet, "/", nil, gin.Param{Key: "jobId", Value: "job-1"})
	handler.GetInstallJobHandler(ctx)
	if response.Code != http.StatusOK {
		t.Fatalf("get status=%d body=%s", response.Code, response.Body.String())
	}

	ctx, response = testContext(http.MethodGet, "/", nil)
	handler.ListInstallJobsHandler(ctx)
	if response.Code != http.StatusOK {
		t.Fatalf("list status=%d body=%s", response.Code, response.Body.String())
	}
}

// Exercises missing route-parameter branches for uninstall and update.
func TestPackageOperationHandlersRequirePackageID(t *testing.T) {
	manager := &stubPackageJobManager{}
	handler := NewPackageInstallHandler(manager)
	for _, invoke := range []func(*gin.Context){
		handler.UninstallPackageHandler,
		handler.UpdatePackageHandler,
	} {
		ctx, response := testContext(http.MethodPost, "/", nil)
		invoke(ctx)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
		}
	}
}

// Exercises successful uninstall and the default internal-error mapping.
func TestUninstallPackageHandlerSuccessAndInternalError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want int
	}{
		{name: "success", want: http.StatusOK},
		{name: "internal", err: errors.New("plain failure"), want: http.StatusInternalServerError},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manager := &stubPackageJobManager{uninstall: func(string) error { return test.err }}
			ctx, response := testContext(http.MethodDelete, "/", nil, gin.Param{Key: "packageId", Value: "demo"})
			NewPackageInstallHandler(manager).UninstallPackageHandler(ctx)
			if response.Code != test.want {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

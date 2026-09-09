package server

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func TestManifestReadinessDefaultsRemainBackwardCompatible(t *testing.T) {
	helmPath := "../../../deployments/helm/agentfield/values.yaml"
	helmTemplatePath := "../../../deployments/helm/agentfield/templates/control-plane-deployment.yaml"
	basePath := "../../../deployments/kubernetes/base/control-plane-deployment.yaml"
	if _, err := os.Stat(helmPath); err != nil {
		t.Skipf("Helm values unavailable in this build context: %v", err)
	}
	if _, err := os.Stat(basePath); err != nil {
		t.Skipf("base manifest unavailable in this build context: %v", err)
	}
	if _, err := os.Stat(helmTemplatePath); err != nil {
		t.Skipf("Helm template unavailable in this build context: %v", err)
	}

	var helm struct {
		ControlPlane struct {
			ReadinessProbe struct {
				Path             string `yaml:"path"`
				PeriodSeconds    int    `yaml:"periodSeconds"`
				FailureThreshold int    `yaml:"failureThreshold"`
			} `yaml:"readinessProbe"`
			ShutdownMinDelay              string `yaml:"shutdownMinDelay"`
			TerminationGracePeriodSeconds int    `yaml:"terminationGracePeriodSeconds"`
		} `yaml:"controlPlane"`
	}
	data, err := os.ReadFile(helmPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := yaml.Unmarshal(data, &helm); err != nil {
		t.Fatal(err)
	}
	templateData, err := os.ReadFile(helmTemplatePath)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name      string
		reference string
		got       any
		want      any
	}{
		{name: "readiness probe path", reference: ".Values.controlPlane.readinessProbe.path", got: helm.ControlPlane.ReadinessProbe.Path, want: "/api/v1/health"},
		{name: "readiness probe period", reference: ".Values.controlPlane.readinessProbe.periodSeconds", got: helm.ControlPlane.ReadinessProbe.PeriodSeconds, want: 2},
		{name: "readiness probe failure threshold", reference: ".Values.controlPlane.readinessProbe.failureThreshold", got: helm.ControlPlane.ReadinessProbe.FailureThreshold, want: 1},
		{name: "shutdown min delay", reference: ".Values.controlPlane.shutdownMinDelay", got: helm.ControlPlane.ShutdownMinDelay, want: "5s"},
		{name: "termination grace period", reference: ".Values.controlPlane.terminationGracePeriodSeconds", got: helm.ControlPlane.TerminationGracePeriodSeconds, want: 60},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.got != test.want {
				t.Fatalf("Helm default = %v, want %v", test.got, test.want)
			}
			if !strings.Contains(string(templateData), test.reference) {
				t.Fatalf("Helm control-plane deployment template does not reference %q", test.reference)
			}
		})
	}
	delay, err := time.ParseDuration(helm.ControlPlane.ShutdownMinDelay)
	if err != nil {
		t.Fatalf("parse Helm shutdown minimum delay: %v", err)
	}
	readinessFailureWindow := time.Duration(helm.ControlPlane.ReadinessProbe.PeriodSeconds*helm.ControlPlane.ReadinessProbe.FailureThreshold) * time.Second
	if readinessFailureWindow >= delay {
		t.Fatalf("readiness failure window %s must be shorter than shutdown minimum delay %s", readinessFailureWindow, delay)
	}

	var base struct {
		Spec struct {
			Template struct {
				Spec struct {
					Containers []struct {
						ReadinessProbe struct {
							HTTPGet struct {
								Path string `yaml:"path"`
							} `yaml:"httpGet"`
							PeriodSeconds    int `yaml:"periodSeconds"`
							FailureThreshold int `yaml:"failureThreshold"`
						} `yaml:"readinessProbe"`
					} `yaml:"containers"`
				} `yaml:"spec"`
			} `yaml:"template"`
		} `yaml:"spec"`
	}
	data, err = os.ReadFile(basePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := yaml.Unmarshal(data, &base); err != nil {
		t.Fatal(err)
	}
	if len(base.Spec.Template.Spec.Containers) == 0 {
		t.Fatal("base manifest has no containers")
	}
	baseReadiness := base.Spec.Template.Spec.Containers[0].ReadinessProbe
	if baseReadiness.HTTPGet.Path != "/api/v1/health" {
		t.Fatalf("base manifest readiness path changed")
	}
	if baseReadiness.PeriodSeconds != 2 || baseReadiness.FailureThreshold != 1 {
		t.Fatalf("base readiness timing = period %ds, threshold %d; want period 2s, threshold 1", baseReadiness.PeriodSeconds, baseReadiness.FailureThreshold)
	}
}

type renderedControlPlane struct {
	TerminationGracePeriodSeconds int
	Container                     renderedControlPlaneContainer
}

type renderedControlPlaneContainer struct {
	Name string `yaml:"name"`
	Env  []struct {
		Name  string `yaml:"name"`
		Value string `yaml:"value"`
	} `yaml:"env"`
	ReadinessProbe struct {
		HTTPGet struct {
			Path string `yaml:"path"`
		} `yaml:"httpGet"`
		PeriodSeconds    int `yaml:"periodSeconds"`
		FailureThreshold int `yaml:"failureThreshold"`
	} `yaml:"readinessProbe"`
}

func TestHelmControlPlaneShutdownContractRenders(t *testing.T) {
	if _, err := exec.LookPath("helm"); err != nil {
		t.Skipf("helm unavailable: %v", err)
	}
	if _, err := os.Stat("../../../deployments/helm/agentfield/Chart.yaml"); err != nil {
		t.Skipf("Helm chart unavailable in this build context: %v", err)
	}

	tests := []struct {
		name              string
		extraArgs         []string
		wantReadinessPath string
		wantMinDelayValue string
		wantMinDelayCount int
	}{
		{name: "defaults", wantReadinessPath: "/api/v1/health", wantMinDelayValue: "5s", wantMinDelayCount: 1},
		{name: "drain aware readiness opt in", extraArgs: []string{"--set-string", "controlPlane.readinessProbe.path=/api/v1/health/ready"}, wantReadinessPath: "/api/v1/health/ready", wantMinDelayValue: "5s", wantMinDelayCount: 1},
		{name: "env map overrides generated value", extraArgs: []string{"--set-string", "controlPlane.env.AGENTFIELD_SHUTDOWN_MIN_DELAY=17s"}, wantReadinessPath: "/api/v1/health", wantMinDelayValue: "17s", wantMinDelayCount: 1},
		{name: "extra env overrides generated value", extraArgs: []string{"--set-string", "controlPlane.extraEnv[0].name=AGENTFIELD_SHUTDOWN_MIN_DELAY", "--set-string", "controlPlane.extraEnv[0].value=23s"}, wantReadinessPath: "/api/v1/health", wantMinDelayValue: "23s", wantMinDelayCount: 1},
		{name: "zero disables generated value", extraArgs: []string{"--set", "controlPlane.shutdownMinDelay=0"}, wantReadinessPath: "/api/v1/health", wantMinDelayCount: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered := renderHelmControlPlane(t, tt.extraArgs...)
			probe := rendered.Container.ReadinessProbe
			if probe.HTTPGet.Path != tt.wantReadinessPath {
				t.Fatalf("readiness path = %q, want %q", probe.HTTPGet.Path, tt.wantReadinessPath)
			}
			if probe.PeriodSeconds != 2 || probe.FailureThreshold != 1 {
				t.Fatalf("readiness timing = period %ds, threshold %d; want period 2s, threshold 1", probe.PeriodSeconds, probe.FailureThreshold)
			}
			if rendered.TerminationGracePeriodSeconds != 60 {
				t.Fatalf("termination grace = %d, want 60", rendered.TerminationGracePeriodSeconds)
			}

			count, value := shutdownMinDelayEnv(rendered.Container)
			if count != tt.wantMinDelayCount {
				t.Fatalf("rendered %d AGENTFIELD_SHUTDOWN_MIN_DELAY entries, want %d", count, tt.wantMinDelayCount)
			}
			if count > 0 && value != tt.wantMinDelayValue {
				t.Fatalf("AGENTFIELD_SHUTDOWN_MIN_DELAY = %q, want %q", value, tt.wantMinDelayValue)
			}
		})
	}
}

func renderHelmControlPlane(t *testing.T, extraArgs ...string) renderedControlPlane {
	t.Helper()
	chartPath := "../../../deployments/helm/agentfield"
	args := append([]string{"template", "shutdown-contract", chartPath}, extraArgs...)
	out, err := exec.Command("helm", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("helm template failed: %v\n%s", err, out)
	}

	decoder := yaml.NewDecoder(bytes.NewReader(out))
	for {
		var manifest struct {
			Kind string `yaml:"kind"`
			Spec struct {
				Template struct {
					Spec struct {
						TerminationGracePeriodSeconds int                             `yaml:"terminationGracePeriodSeconds"`
						Containers                    []renderedControlPlaneContainer `yaml:"containers"`
					} `yaml:"spec"`
				} `yaml:"template"`
			} `yaml:"spec"`
		}
		if err := decoder.Decode(&manifest); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("decode helm output: %v", err)
		}
		if manifest.Kind != "Deployment" {
			continue
		}
		for _, container := range manifest.Spec.Template.Spec.Containers {
			if container.Name == "control-plane" {
				return renderedControlPlane{
					TerminationGracePeriodSeconds: manifest.Spec.Template.Spec.TerminationGracePeriodSeconds,
					Container:                     container,
				}
			}
		}
	}
	t.Fatal("rendered chart has no control-plane Deployment container")
	return renderedControlPlane{}
}

func shutdownMinDelayEnv(container renderedControlPlaneContainer) (int, string) {
	count := 0
	value := ""
	for _, env := range container.Env {
		if env.Name == "AGENTFIELD_SHUTDOWN_MIN_DELAY" {
			count++
			value = env.Value
		}
	}
	return count, value
}

package server

import (
	"os"
	"strings"
	"testing"

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
				Path string `yaml:"path"`
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

	var base struct {
		Spec struct {
			Template struct {
				Spec struct {
					Containers []struct {
						ReadinessProbe struct {
							HTTPGet struct {
								Path string `yaml:"path"`
							} `yaml:"httpGet"`
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
	if len(base.Spec.Template.Spec.Containers) == 0 || base.Spec.Template.Spec.Containers[0].ReadinessProbe.HTTPGet.Path != "/api/v1/health" {
		t.Fatalf("base manifest readiness path changed")
	}
}

package server

import (
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestManifestReadinessDefaultsRemainBackwardCompatible(t *testing.T) {
	helmPath := "../../../deployments/helm/agentfield/values.yaml"
	basePath := "../../../deployments/kubernetes/base/control-plane-deployment.yaml"
	if _, err := os.Stat(helmPath); err != nil {
		t.Skipf("Helm values unavailable in this build context: %v", err)
	}
	if _, err := os.Stat(basePath); err != nil {
		t.Skipf("base manifest unavailable in this build context: %v", err)
	}

	var helm struct {
		ControlPlane struct {
			ReadinessProbe struct {
				Path string `yaml:"path"`
			} `yaml:"readinessProbe"`
		} `yaml:"controlPlane"`
	}
	data, err := os.ReadFile(helmPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := yaml.Unmarshal(data, &helm); err != nil {
		t.Fatal(err)
	}
	if helm.ControlPlane.ReadinessProbe.Path != "/api/v1/health" {
		t.Fatalf("Helm readiness path = %q", helm.ControlPlane.ReadinessProbe.Path)
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

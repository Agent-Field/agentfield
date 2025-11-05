package server

import (
	"context"
	"fmt"

	"github.com/your-org/haxen/control-plane/pkg/types"
)

type stubPackageStorage struct {
	packages map[string]*types.AgentPackage
	getCalls []string
}

func newStubPackageStorage() *stubPackageStorage {
	return &stubPackageStorage{packages: make(map[string]*types.AgentPackage)}
}

func (s *stubPackageStorage) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	s.getCalls = append(s.getCalls, packageID)
	if pkg, ok := s.packages[packageID]; ok {
		return pkg, nil
	}
	return nil, fmt.Errorf("package %s not found", packageID)
}

func (s *stubPackageStorage) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	s.packages[pkg.ID] = pkg
	return nil
}

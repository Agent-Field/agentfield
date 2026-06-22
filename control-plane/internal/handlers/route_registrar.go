package handlers

import "github.com/gin-gonic/gin"

// RouteRegistrar is implemented by handler types that register their HTTP
// routes on a Gin router. Using the gin.IRouter interface (rather than the
// concrete *gin.RouterGroup) allows handlers to be composed polymorphically
// and tested with lightweight router mocks.
type RouteRegistrar interface {
	RegisterRoutes(r gin.IRouter)
}

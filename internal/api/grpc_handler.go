package api

import (
	"context"

	pb "github.com/D13ya/DaZZLeD/api/proto/v1"
	"github.com/D13ya/DaZZLeD/internal/app"
)

type GRPCHandler struct {
	pb.UnimplementedAuthorityServiceServer
	service *app.ServerService
}

func NewGRPCHandler(service *app.ServerService) *GRPCHandler {
	return &GRPCHandler{service: service}
}

func (h *GRPCHandler) CheckImage(ctx context.Context, req *pb.BlindCheckRequest) (*pb.BlindCheckResponse, error) {
	return h.service.HandleCheckImage(ctx, req)
}

func (h *GRPCHandler) UploadShare(ctx context.Context, req *pb.VoucherShare) (*pb.ShareResponse, error) {
	return h.service.HandleUploadShare(ctx, req)
}

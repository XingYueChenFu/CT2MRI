import torch

def smooothing_loss3D(y_pred):
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    dy = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dx = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])

    dx = dx * dx
    dy = dy * dy
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d
    return d

def smooothing_loss2D(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d
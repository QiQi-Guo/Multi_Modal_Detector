import math
import time

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmdet3d.models.builder import VTRANSFORMS
import copy

__all__ = ["fast_BEV"]


@VTRANSFORMS.register_module()
class fast_BEV(nn.Module):
    def __init__(self,
                 image_size,
                 voxel_size,
                 n_voxels,
                 flag=0,
                 voxel_center=[0.0, 0.0, -1.7]
                 ) -> None:
        super().__init__()
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.n_voxels = n_voxels
        self.final_conv = nn.Conv2d(256, 80, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(80)
        self.relu = nn.ReLU(inplace=True)
        self.flag = flag
        self.voxel_center = voxel_center

        self.trans_bias = 0.4  # m   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.rot_bias = 2.0  # deg   [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        self.T_matrix_bias = False


    def forward(self,
                mlvl_feats,  # img, (b, num, c, h, w)
                ori_points,
                img,
                lidar2image,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                mask=None):

        stride_i = math.ceil(self.image_size[-1] / mlvl_feats.shape[-1])

        volumes = []
        for i in range(mlvl_feats.shape[0]):
            feat_i = mlvl_feats[i]  # [nv, c, h, w]
            # point: [3, vx, vy, vz]
            point = get_points(torch.tensor(self.n_voxels), torch.tensor(self.voxel_size), self.voxel_center).to(feat_i.device)

            n_x_voxels, n_y_voxels, n_z_voxels = point.shape[-3:]
            n_images, n_channels, height, width = feat_i.shape

            lidar_aug_m = torch.tensor(lidar_aug_matrix[i]).to(feat_i.device)
            lidar_aug_m_t = copy.deepcopy(lidar_aug_m[:3, -1]).to(feat_i.device)
            lidar_aug_m_r = copy.deepcopy(lidar_aug_m[:3, :3]).to(feat_i.device)
            img_aug_m = torch.tensor(img_aug_matrix[i]).to(feat_i.device)
            img_aug_m_t = copy.deepcopy(img_aug_m[..., -1]).to(feat_i.device)
            img_aug_m_r = copy.deepcopy(img_aug_m).to(feat_i.device)
            img_aug_m_r[:, :-1, -1] = 0
            lidar2image_m = torch.tensor(lidar2image[i]).to(feat_i.device)
            projection = (img_aug_m_r @ lidar2image_m)
            projection = projection[:, :3, :]

            point = point.view(1, 3, -1)
            point -= lidar_aug_m_t.unsqueeze(0).unsqueeze(-1)
            point = torch.bmm(lidar_aug_m_r.T.unsqueeze(0), point)

            if self.T_matrix_bias:
                trans_bias = torch.tensor((np.random.rand(1, 3) - 0.5) / (0.5 / self.trans_bias)).to(feat_i.device)
                rot_bias_deg = (np.random.rand(3) - 0.5) * (self.rot_bias / 0.5)
                rot_bias_rad = rot_bias_deg * 3.14 / 180
                rot_bias_M = torch.tensor(GetRotationMatrix(rot_bias_rad), dtype=torch.float32).to(feat_i.device)
                point -= trans_bias.unsqueeze(-1)
                point = torch.bmm(rot_bias_M.unsqueeze(0), point)

            point = torch.cat((point, torch.ones_like(point[:, :1])), dim=1).expand(n_images, 4, -1)
            point_to_img = torch.bmm(projection, point)

            Zu_img = point_to_img[:, 0]  # Z*u_img  [6, 180*180*4]
            Zv_img = point_to_img[:, 1]  # Z*v_img  [6, 180*180*4]
            Z = point_to_img[:, 2]  # [6, 180*180*4]

            u_img = Zu_img / Z  # [6, 180*180*4]  # horizontal coord in img
            v_img = Zv_img / Z  # [6, 180*180*4]  # vertical coord in img
            u_img += img_aug_m_t[..., 0].unsqueeze(-1)
            v_img += img_aug_m_t[..., 1].unsqueeze(-1)

            u_fm = (u_img / stride_i).round().long()  # u:W  stride = img/feature_map
            v_fm = (v_img / stride_i).round().long()  # v:H  stride = img/feature_map
            valid = (u_fm >= 0) & (v_fm >= 0) & (u_fm < width) & (v_fm < height) & (Z > 0)  # valid point mask [6, 180*180*4] bool

            volume = torch.zeros((n_channels, point.shape[-1]), device=feat_i.device).type_as(feat_i)  # [256, 180*180*4]
            for j in range(n_images):
                volume[:, valid[j]] = feat_i[j, :, v_fm[j, valid[j]], u_fm[j, valid[j]]]
            volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
            volumes.append(volume)

        volume_list = torch.stack(volumes)
        volume_list = volume_list.sum(dim=-1, keepdim=True)
        final_1 = torch.cat(volume_list.unbind(dim=4), 1)
        final_1 = self.relu(self.bn(self.final_conv(final_1)))
        return final_1


@torch.no_grad()
def get_points(n_voxels, voxel_size, voxel_center=[0.0, 0.0, -1.0]):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = torch.Tensor(voxel_center) - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def GetRotationMatrix(rot):
    theta_x = rot[0]
    theta_y = rot[1]
    theta_z = rot[2]
    sx = np.sin(theta_x)
    cx = np.cos(theta_x)
    sy = np.sin(theta_y)
    cy = np.cos(theta_y)
    sz = np.sin(theta_z)
    cz = np.cos(theta_z)
    return np.array([
              [cy*cz, cz*sx*sy-cx*sz, sx*sz+cx*cz*sy],
              [cy*sz, cx*cz+sx*sy*sz, cx*sy*sz-cz*sz],
              [-sy, cy*sx, cx*cy]])
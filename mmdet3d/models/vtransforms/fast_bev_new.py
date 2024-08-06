import math
import time

import torch
import torch.nn as nn

from mmdet3d.models.builder import VTRANSFORMS
from mmdet3d.models.builder import build_neck
import copy

__all__ = ["fast_BEV"]


@VTRANSFORMS.register_module()
class fast_BEV(nn.Module):
    def __init__(self,
                 image_size,
                 voxel_size,
                 n_voxels
                 ) -> None:
        super().__init__()
        self.image_size = image_size
        voxel_grid_np = self.get_voxel_grid(n_voxels=torch.tensor(n_voxels), voxel_size=torch.tensor(voxel_size)).numpy()
        self.voxel_grid = nn.Parameter(torch.from_numpy(voxel_grid_np), requires_grad=True)  # voxel_grid_np: [3, vx, vy, vz]

        self.final_conv = nn.Conv2d(1024, 80, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(80)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,
                img_features,  # img, (b, num, c, h, w)
                lidar2image,
                img_aug_matrix,
                lidar_aug_matrix,
                point_clouds,
                camera2ego,
                lidar2ego,
                cam_intrinsic,
                cam_2_lidar,
                img_metas
                ):
        stride_i = math.ceil(self.image_size[-1] / img_features.shape[-1])

        volumes = []
        for i in range(img_features.shape[0]):
            feat_i = img_features[i]  # [nv, c, h, w]
            projection = self.compute_projection_with_aug(img_aug_matrix[i], lidar_aug_matrix[i], lidar2image[i]).to(feat_i.device)
            volume = self.backproject_inplace(feat_i, self.voxel_grid, projection, stride_i)  # [c, vx, vy, vz]
            volumes.append(volume)

        volume_list = torch.stack(volumes)  # list([bs, c, vx, vy, vz])
        final_1 = torch.cat(volume_list.unbind(dim=4), 1)
        final_1 = self.relu(self.bn(self.final_conv(final_1)))
        return final_1


    def compute_projection_with_aug(self, img_aug_matrix, lidar_aug_matrix, lidar2image):
        img_aug_matrix = torch.tensor(img_aug_matrix)
        lidar_aug_matrix = torch.tensor(lidar_aug_matrix)
        projection = (img_aug_matrix @ lidar2image @ lidar_aug_matrix)
        projection = projection[:, :3, :]
        return projection


    def backproject_inplace(self, features, voxel_grid, projection, stride):
        n_images, n_channels, height, width = features.shape
        n_x_voxels, n_y_voxels, n_z_voxels = voxel_grid.shape[-3:]
        voxel_grid = voxel_grid.view(1, 3, -1).expand(n_images, 3, -1)  # [3, 180, 180, 4] -> [1, 3, 180*180*4] -> [6, 3, 180*180*4]
        voxel_grid = torch.cat((voxel_grid, torch.ones_like(voxel_grid[:, :1])), dim=1)  # [X, Y, Z] -> [X, Y, Z, 1]
        points_to_img = torch.bmm(projection, voxel_grid)  # lidar2img   T * [X, Y, Z, 1]  ->  [6, 3, 4] * [6, 4, 180*180*4]  ->  [6, 3, 180*180*4]
        Zu_img = points_to_img[:, 0]  # Z*u_img  [6, 180*180*4]
        Zv_img = points_to_img[:, 1]  # Z*v_img  [6, 180*180*4]
        Z = points_to_img[:, 2]  # [6, 180*180*4]

        u_img = Zu_img / Z  # [6, 180*180*4]  # horizontal coord in img
        v_img = Zv_img / Z  # [6, 180*180*4]  # vertical coord in img
        u_fm = (u_img / stride).round().long()  # u:W  stride = img/feature_map
        v_fm = (v_img / stride).round().long()  # v:H  stride = img/feature_map
        valid = (u_fm >= 0) & (v_fm >= 0) & (u_fm < width) & (v_fm < height) & (Z > 0)  # valid point mask [6, 180*180*4] bool

        volume = torch.zeros((n_channels, voxel_grid.shape[-1]), device=features.device).type_as(features)  # [256, 180*180*4]
        for i in range(n_images):
            volume[:, valid[i]] = features[i, :, v_fm[i, valid[i]], u_fm[i, valid[i]]]
        volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)

        return volume


    @torch.no_grad()
    def get_voxel_grid(self, n_voxels, voxel_size):
        voxel_grid = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(n_voxels[0]),
                    torch.arange(n_voxels[1]),
                    torch.arange(n_voxels[2]),
                ]
            )
        )
        new_origin = torch.Tensor([0., 0., -1]) - n_voxels / 2.0 * voxel_size
        voxel_grid = voxel_grid * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
        return voxel_grid

import math
import time

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmdet3d.models.builder import VTRANSFORMS
from mmdet3d.models.builder import build_neck
import copy


__all__ = ["fast_BEV"]

@VTRANSFORMS.register_module()
class fast_BEV(nn.Module):
    def __init__(self,
        # xbound,
        # ybound,
        image_size,
        voxel_size,
        n_voxels,
        flag=0,
    ) -> None:
        super().__init__()
        # image_size =[256, 704],
        # xbound = xbound,
        # ybound = xbound,
        # )
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.n_voxels = n_voxels
        self.final_conv = nn.Conv2d(256, 80, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(80)
        self.relu = nn.ReLU(inplace=True)
        self.flag = flag


    def forward(self,
        # lidar2camera, #lidar2img ex (b, num, 4, 4)
        mlvl_feats,  # img, (b, num, c, h, w)
        points,
        ori_points,
        img,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        cam_2_lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas):

        stride_i = math.ceil(self.image_size[-1] / mlvl_feats.shape[-1])

        volumes =[]
        for i in range(mlvl_feats.shape[0]):
            feat_i = mlvl_feats[i]  # [nv, c, h, w]
            projection = compute_projection_with_aug(img_aug_matrix[i], lidar_aug_matrix[i], lidar2image[i]).to(feat_i.device)
            img_aug_matrix_i_t = img_aug_matrix[i][..., -1]

            # ori_point_i = ori_points[i][:, :3].permute(1, 0).expand(6, 3, -1)
            # img_i = img[i]
            # ori_point_i = torch.cat((ori_point_i, torch.ones_like(ori_point_i[:, :1])), dim=1)  # [X, Y, Z] -> [X, Y, Z, 1]
            # points_to_img_i = torch.bmm(projection, ori_point_i)
            # Zu_img = points_to_img_i[:, 0]  # Z*u_img  [6, 180*180*4]
            # Zv_img = points_to_img_i[:, 1]  # Z*v_img  [6, 180*180*4]
            # Z = points_to_img_i[:, 2]  # [6, 180*180*4]
            #
            # u_img = Zu_img / Z  # [6, 180*180*4]  # horizontal coord in img
            # v_img = Zv_img / Z  # [6, 180*180*4]  # vertical coord in img
            # u_img += img_aug_matrix_i_t[..., 0].unsqueeze(-1)
            # v_img += img_aug_matrix_i_t[..., 1].unsqueeze(-1)
            # u_img = u_img.round().long().detach().cpu()
            # v_img = v_img.round().long().detach().cpu()
            # valid = (u_img >= 0) & (v_img >= 0) & (u_img < 704) & (v_img < 256) & (Z.detach().cpu() > 0)
            #
            # img_cloth = np.zeros_like(img_i)
            # for h in range(6):
            #     for j in range(valid.shape[1]):
            #         if valid[h, j]:
            #             img_cloth[h, v_img[h, j], u_img[h, j], 1] = 1
            #     cv2.imshow('img_cloth', img_cloth[h] + img_i[h]*0.3)
            #     cv2.waitKey()


            n_voxel = self.n_voxels
            voxel_size = self.voxel_size
            # points: [3, vx, vy, vz]
            point = get_points(n_voxels=torch.tensor(n_voxel), voxel_size=torch.tensor(voxel_size)).to(feat_i.device)
            volume = backproject_inplace(feat_i, point, projection, img_aug_matrix_i_t, stride_i)  # [c, vx, vy, vz]
            volumes.append(volume)

        volume_list = torch.stack(volumes) * points.permute(0, 2, 3, 1).unsqueeze(1) # list([bs, c, vx, vy, vz])
        # volume_list = torch.stack(volumes)
        volume_list = volume_list.sum(dim=-1, keepdim=True)
        final_1 = torch.cat(volume_list.unbind(dim=4), 1)
        final_1 = self.relu(self.bn(self.final_conv(final_1)))
        return final_1



def compute_projection_with_aug(img_aug_matrix, lidar_aug_matrix, lidar2image):
    img_aug_matrix_r = torch.tensor(img_aug_matrix)
    img_aug_matrix_r[..., -1] = 0
    lidar_aug_matrix = torch.tensor(lidar_aug_matrix)
    projection = (img_aug_matrix_r @ lidar2image @ lidar_aug_matrix)
    projection = projection[:, :3, :]
    return projection


@torch.no_grad()
def get_points(n_voxels, voxel_size):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = torch.Tensor([0., 0., -1]) - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

def backproject_inplace(features, points, projection, img_aug_matrix_i_t, stride):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)  # [3, 180, 180, 4] -> [1, 3, 180*180*4] -> [6, 3, 180*180*4]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)  # [X, Y, Z] -> [X, Y, Z, 1]
    points_to_img = torch.bmm(projection, points)  # lidar2img   T * [X, Y, Z, 1]  ->  [6, 3, 4] * [6, 4, 180*180*4]  ->  [6, 3, 180*180*4]
    Zu_img = points_to_img[:, 0]  # Z*u_img  [6, 180*180*4]
    Zv_img = points_to_img[:, 1]  # Z*v_img  [6, 180*180*4]
    Z = points_to_img[:, 2]  # [6, 180*180*4]

    u_img = Zu_img / Z  # [6, 180*180*4]  # horizontal coord in img
    v_img = Zv_img / Z  # [6, 180*180*4]  # vertical coord in img
    u_img += img_aug_matrix_i_t[..., 0].unsqueeze(-1)
    v_img += img_aug_matrix_i_t[..., 1].unsqueeze(-1)

    u_fm = (u_img/stride).round().long()  # u:W  stride = img/feature_map
    v_fm = (v_img/stride).round().long()  # v:H  stride = img/feature_map
    valid = (u_fm >= 0) & (v_fm >= 0) & (u_fm < width) & (v_fm < height) & (Z > 0)  # valid point mask [6, 180*180*4] bool

    volume = torch.zeros((n_channels, points.shape[-1]), device=features.device).type_as(features)  # [256, 180*180*4]
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, v_fm[i, valid[i]], u_fm[i, valid[i]]]
    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)

    return volume




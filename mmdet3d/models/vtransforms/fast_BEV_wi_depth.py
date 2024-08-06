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
                 image_size,
                 voxel_size,
                 n_voxels,
                 voxel_center=[0., 0., -1.7],
                 flag=0,
                 ) -> None:
        super().__init__()
        self.voxel_center = voxel_center
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.n_voxels = n_voxels
        self.final_conv = nn.Conv2d(256, 80, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(80)
        self.relu = nn.ReLU(inplace=True)
        self.flag = flag
        self.depth_conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.fuse_conv_layers = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.self_att_layer = nn.Sequential(
            nn.Conv2d(80, 1, 1, padding=0),
            nn.Sigmoid(),
        )

        self.visualize_times = 100
        self.visualize_count = 0
        self.calib_bias = False

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
        batch_size = mlvl_feats.shape[0]

        volumes = []
        for i in range(mlvl_feats.shape[0]):
            feat_i = mlvl_feats[i]  # [nv, c, h, w]
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

            trans_bias = torch.tensor((np.random.rand(1, 3) - 0.5) / 1.25).to(feat_i.device)
            rot_bias_deg = (np.random.randn(3)*0.5) + 2
            # rot_bias_deg = (np.random.rand(3) - 0.5) * 5
            rot_bias_rad = rot_bias_deg * 3.14 / 180
            rot_bias_M = torch.tensor(GetRotationMatrix(rot_bias_rad), dtype=torch.float32).to(feat_i.device)

            depth = torch.zeros((6, 1, *self.image_size), dtype=torch.float16).to(mlvl_feats.device)

            ori_point_i = ori_points[i][:, :3].permute(1, 0).unsqueeze(0)
            ori_point_i -= lidar_aug_m_t.unsqueeze(0).unsqueeze(-1)
            ori_point_i = torch.bmm(lidar_aug_m_r.T.unsqueeze(0), ori_point_i)
            if self.calib_bias:
                ori_point_i -= trans_bias.unsqueeze(-1)
                ori_point_i = torch.bmm(rot_bias_M.unsqueeze(0), ori_point_i)

            ori_point_i = torch.cat((ori_point_i, torch.ones_like(ori_point_i[:, :1])), dim=1).expand(n_images, 4, -1)
            points_to_img_i = torch.bmm(projection, ori_point_i)  # [6, 3, p_n]
            Zu_img = points_to_img_i[:, 0]  # Z*u_img  [6, p_n]
            Zv_img = points_to_img_i[:, 1]  # Z*v_img  [6, p_n]
            Z = points_to_img_i[:, 2]  # [6, 180*180*4]

            u_img = Zu_img / Z  # [6, 180*180*4]  # horizontal coord in img
            v_img = Zv_img / Z  # [6, 180*180*4]  # vertical coord in img
            u_img += img_aug_m_t[..., 0].unsqueeze(-1)
            v_img += img_aug_m_t[..., 1].unsqueeze(-1)

            dist = points_to_img_i[:, 2, :]  # Z axis
            on_img = (u_img >= 0) & (v_img >= 0) & (u_img < 704) & (v_img < 256) & (Z.detach() > 0)
            for c in range(on_img.shape[0]):
                masked_coords_u = u_img[c, on_img[c]].long()  # img_coord for the reserved points
                masked_coords_v = v_img[c, on_img[c]].long()  # img_coord for the reserved points
                masked_dist = dist[c, on_img[c]]  # Z axis of the reserved points
                depth[c, 0, masked_coords_v, masked_coords_u] = masked_dist

            # for h in range(6):
            #     depth_img = depth[h, 0].unsqueeze(-1).detach().cpu().numpy().astype(np.float)
            #     depth_img = depth_img/depth_img.max()
            #     cv2.imshow('img_cloth', depth_img + img[i, h] * 0.3)
            #     cv2.waitKey()


            depth_feat_i = self.depth_conv_layers(depth)
            fuse_feat_i = torch.cat([depth_feat_i, feat_i], dim=1)
            fuse_feat_i = self.fuse_conv_layers(fuse_feat_i)



            # point: [3, vx, vy, vz]
            point = get_points(torch.tensor(self.n_voxels), torch.tensor(self.voxel_size), self.voxel_center).to(
                feat_i.device)
            n_x_voxels, n_y_voxels, n_z_voxels = point.shape[-3:]
            point = point.view(1, 3, -1)
            point -= lidar_aug_m_t.unsqueeze(0).unsqueeze(-1)
            point = torch.bmm(lidar_aug_m_r.T.unsqueeze(0), point)
            if self.calib_bias:
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
                volume[:, valid[j]] = fuse_feat_i[j, :, v_fm[j, valid[j]], u_fm[j, valid[j]]]
            volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
            volumes.append(volume)

        volume_list = torch.stack(volumes)
        volume_list = volume_list.sum(dim=-1, keepdim=True)
        final_1 = torch.cat(volume_list.unbind(dim=4), 1)
        final_1 = self.relu(self.bn(self.final_conv(final_1)))
        final_mask = self.self_att_layer(final_1)
        final_1 = final_mask * final_1
        return final_1


@torch.no_grad()
def get_points(n_voxels, voxel_size, voxel_center=[0., 0., -1.7]):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = torch.Tensor([0., 0., -1.7]) - n_voxels / 2.0 * voxel_size
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



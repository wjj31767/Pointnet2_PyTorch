from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

def knn_point(k,xyz1,xyz2):
    b,n,c =xyz1.shape
    _,m,_ = xyz2.shape
    xyz1 = xyz1.view(b,1,n,c).repeat(1,m,1,1)
    xyz2 = xyz2.view(b,m,1,c).repeat(1,1,n,1)
    dist = torch.sum((xyz1-xyz2)**2,-1)
    # print("KNNN",dist.shape,k)
    val,idx = torch.topk(dist,k=k)
    # print(xyz1.shape,xyz2.shape,dist.shape,val.shape,idx.shape)
    return (val,idx)

def group(xyz, points, k, dilation=1, use_xyz=False):
    _, idx = knn_point(k*dilation+1, xyz, xyz)
    idx = idx[:, :, 1::dilation].int().contiguous()
    # print("xyz",xyz.shape,idx.shape)
    xyz_trans = xyz.transpose(1, 2).contiguous()
    grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx.int().permute(0,2,1).contiguous())  # (batch_size, npoint, k, 3)
    # print("GRO",grouped_xyz.shape,xyz.shape)
    grouped_xyz -= torch.unsqueeze(xyz_trans, 2)  # translation normalization
    if points is not None:
        # print(points.shape)
        grouped_points = pointnet2_utils.grouping_operation(points.permute(0,2,1).contiguous(), idx.int().permute(0,2,1).contiguous())  # (batch_size, npoint, k, channel)
        if use_xyz:
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (batch_size, npoint, k, 3+channel)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


def pool(xyz, points, k, npoint):
    xyz_flipped = xyz.transpose(1, 2).contiguous()
    new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.furthest_point_sample(xyz_flipped, npoint)).transpose(1,2).contiguous()
    _, idx = knn_point(k, xyz, new_xyz)
    new_points = torch.max(pointnet2_utils.grouping_operation(points.permute(0,2,1).contiguous(), idx.int().permute(0,2,1).contiguous()).permute(0,3,2,1), dim=2).values

    return new_xyz, new_points

class pointcnn(nn.Module):
    def __init__(self,k,n_cout,n_blocks,activation=F.relu,bn=False):
        super(pointcnn,self).__init__()
        self.bn=bn
        self.k = k
        self.n_cout=n_cout
        self.n_blocks = n_blocks
        self.activation = activation
        self.conv1 = nn.Conv2d(3,n_cout,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(n_cout,n_cout,kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(n_cout)
    def forward(self,xyz):
        # grouped_points: knn points coordinates (normalized: minus centual points)
        # print(xyz.shape)
        xyz = xyz.permute(0,2,1)
        _, grouped_points, _ = group(xyz, None, self.k)
        # print('n_blocks: ', n_blocks)
        # print('is_training: ', is_training)

        for idx in range(self.n_blocks):
            if idx==0:
                grouped_points = self.conv1(grouped_points)
            else:
                grouped_points = self.conv2(grouped_points)
            if idx == self.n_blocks - 1:
                return torch.max(grouped_points, 2).values
            else:
                if self.bn:
                    grouped_points = self.bn(grouped_points)
                grouped_points = self.activation(grouped_points)

class res_gcn_d(nn.Module):
    def __init__(self, k, n_cout, n_blocks,indices=None):
        super(res_gcn_d,self).__init__()
        self.k=k
        self.n_blocks = n_blocks
        self.indices = indices
        self.convs = nn.ModuleList()
        self.n_cout = n_cout
        for i in range(n_blocks):
            self.convs.append(nn.Conv2d(n_cout,n_cout,kernel_size=(1,1)))
            self.convs.append(nn.Conv2d(n_cout,n_cout,kernel_size=(1,1)))

    def forward(self,xyz,points):
        for idx in range(self.n_blocks):
            shortcut = points
            # Center Features
            points = points.permute(0, 2, 1).contiguous()
            xyz = xyz.permute(0, 2, 1).contiguous()
            points = F.leaky_relu(points)
            # Neighbor Features
            if idx == 0 and self.indices is None:

                _, grouped_points, indices = group(xyz, points, self.k)
            else:
                grouped_points = torch.unsqueeze(points,dim=2)
                grouped_points = grouped_points.permute(0,3,2,1).contiguous()

            # Center Conv
            points = points.permute(0, 2, 1).contiguous()
            xyz = xyz.permute(0, 2, 1).contiguous()
            center_points = torch.unsqueeze(points, dim=2)
            points = self.convs[2*idx](center_points)
            # Neighbor Conv
            # print("RES_GCN_D",grouped_points.shape)
            # grouped_points = grouped_points.permute(0,3,2,1).contiguous()
            grouped_points_nn = self.convs[2*idx+1](grouped_points)
            # CNN
            points = torch.mean(torch.cat([points, grouped_points_nn], dim=2), dim=2) + shortcut

        return points

class res_gcn_d_last(nn.Module):
    def __init__(self, n_cout,in_channel=64):
        super(res_gcn_d_last, self).__init__()
        self.conv = nn.Conv2d(in_channel,n_cout,kernel_size=(1,1))
    def forward(self,points):
        points = F.leaky_relu(points)
        center_points = torch.unsqueeze(points, dim=2)
        points = torch.squeeze(self.conv(center_points), dim=2)
        points = points.permute(0,2,1)
        return points


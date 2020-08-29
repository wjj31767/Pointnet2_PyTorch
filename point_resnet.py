import math
import sys
from pointnet2.models.res_gcn_module import pointcnn, knn_point, pool, res_gcn_d, res_gcn_d_last
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class Discriminator(nn.Module):
    def __init__(self,activation,num_point):
        super(Discriminator,self).__init__()
        self.activation = activation
        self.pointcnn = pointcnn(8,64,2,activation=self.activation)
        self.block_num = int(math.log2(num_point / 64) / 2)
        self.res_gcn_d_list = nn.ModuleList()
        for i in range(self.block_num):
            self.res_gcn_d_list.append(res_gcn_d(8,64,4))
        self.res_gcn_d_last = res_gcn_d_last(1)
    def forward(self,xyz):
        # if xyz.shape[0]>3:
        #     points = xyz[:,3:,:]
        # else:
        points = self.pointcnn(xyz)

        for i in range(self.block_num):
            points = points.permute(0,2,1).contiguous()
            xyz = xyz.permute(0,2,1).contiguous()
            # print("DISS",xyz.shape,points.shape)
            xyz, points = pool(xyz, points, 8, points.shape[1] // 4)
            points = points.permute(0, 2, 1)
            xyz = xyz.permute(0, 2, 1)
            # print("AFTER POOL",xyz.shape,points.shape)

            points = self.res_gcn_d_list[i](xyz, points)
        points = self.res_gcn_d_last(points)

        return points
if __name__== '__main__':
    net = Discriminator(activation=nn.functional.leaky_relu)
    for name, param in net.named_parameters():
        print(name, param.size())
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule


class Generator(nn.Module):
    def __init__(self, num_points=2048, bradius=1.0, up_ratio=4,bn=False):
        super(Generator, self).__init__()
        self.num_points = num_points
        self.up_ratio = up_ratio
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=num_points,
                radius=bradius*0.05,
                nsample=32,
                mlp=[3, 32, 32, 64],
                bn=False,
                # use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=num_points//2,
                radius=bradius*0.1,
                nsample=32,
                mlp=[64, 64, 64, 128],
                bn=False,
                # use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=num_points//4,
                radius=bradius*0.2,
                nsample=32,
                mlp=[128, 128, 128, 256],
                bn=False,
                # use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=num_points//8,
                radius=bradius*0.3,
                nsample=32,
                mlp=[256, 256, 256, 512],
                bn=False,
                # use_xyz=self.hparams["model.use_xyz"],
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[512, 64],bn=False))
        self.FP_modules.append(PointnetFPModule(mlp=[256, 64],bn=False))
        self.FP_modules.append(PointnetFPModule(mlp=[128, 64],bn=False))
        self.New_points_list = nn.ModuleList()
        for i in range(self.up_ratio):
            self.New_points_list.append(
                nn.Sequential(
                    nn.Conv2d(259, 256, kernel_size=1, bias=False),
                    # nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    # nn.Dropout(0.5),
                    nn.Conv2d(256, 128, kernel_size=1),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    # nn.Dropout(0.5),
                )
            )
        self.fc_lyaer = nn.Sequential(
            nn.Conv2d(168, 64, kernel_size=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv2d(64, 3, kernel_size=1),
            # nn.BatchNorm2d(3),
            nn.ReLU(True),
            # nn.Dropout(0.5),
        )
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
    def forward(self, pointcloud,labels_onehot):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        l1_xyz, l1_points = self.SA_modules[0](xyz, features)
        l2_xyz, l2_points = self.SA_modules[1](l1_xyz, l1_points)
        l3_xyz, l3_points = self.SA_modules[2](l2_xyz, l2_points)
        l4_xyz, l4_points = self.SA_modules[3](l3_xyz, l3_points)
        up_l4_points = self.FP_modules[0](xyz, l4_xyz, None, l4_points)
        up_l3_points = self.FP_modules[1](xyz, l3_xyz, None, l3_points)
        up_l2_points = self.FP_modules[2](xyz, l2_xyz, None, l2_points)
        new_points_list = []
        for i in range(self.up_ratio):
            concat_feat = torch.cat([up_l4_points, up_l3_points, up_l2_points, l1_points, xyz.permute(0,2,1)], dim=1)
            concat_feat = torch.unsqueeze(concat_feat, dim=2)
            new_points = self.New_points_list[i](concat_feat)
            new_points_list.append(new_points)

        net = torch.cat(new_points_list, dim=1)
        # print(labels_onehot.shape)
        labels_onehot = labels_onehot.float()
        labels_onehot = torch.unsqueeze(labels_onehot, 2)
        labels_onehot = torch.unsqueeze(labels_onehot, 2)
        labels_onehot = labels_onehot.repeat(1, 1, 1, self.num_points)
        # print(labels_onehot.shape,net.shape)
        net = torch.cat([net, labels_onehot], 1)
        coord = self.fc_lyaer(net)
        coord = torch.squeeze(coord,2)
        return coord


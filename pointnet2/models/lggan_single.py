import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from pointnet2.models.lgnet import Generator
from pointnet2.models.point_resnet import Discriminator
from pointnet2.models.pointnet import PointNetCls
import pointnet2.data.data_utils as d_utils
from pointnet2.data.ModelNet40Loader import ModelNet40Cls
import numpy as np
from pytorch_lightning.trainer import Trainer
from collections import OrderedDict
def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        super(GAN,self).__init__()

        self.hparams = hparams
        self.TAU = 1e2
        self.generator = Generator(up_ratio=1)
        self.discrimitor = Discriminator(torch.nn.functional.leaky_relu,num_point=2048)
        self.attack_model = PointNetCls(40,bool(1))
        checkpoint = torch.load('/home/wei/Pointnet2_PyTorch/pointnet2/models/checkpoints/example1.pth')
        self.attack_model.load_state_dict(checkpoint['model_state_dict'])
        self.attack_model.eval()
        self.attack_model_criterion = F.cross_entropy
        self.total = 0
        self.correct = 0
        self.correct_adv = 0
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud,labels_onehot):

        return self.generator(pointcloud,labels_onehot)

    def _generate_labels(self,labels):
        targets = torch.randint(0,40,labels.shape)
        for i in range(len(targets)):
            while labels[i] == targets[i]:
                targets[i] = torch.randint(0, 40,(1,))

        return targets

    def one_hot(self,x, n_class, dtype=torch.float32):
        # X shape: (batch), output shape: (batch, n_class)
        x = x.long()
        res = torch.zeros(x.shape[0], n_class, dtype=dtype).cuda()
        res.scatter_(1, x.view(-1, 1), 1)
        return res

    def training_step(self, batch, batch_idx, optimizer_idx):
        pc, labels = batch
        target_labels = self._generate_labels(labels)
        target_labels = target_labels.long().cuda()
        target_labelsg = self.one_hot(target_labels, 40).cuda()
        if optimizer_idx == 0:
            points_adv = self.forward(pc, target_labelsg)
            # print(points_adv.shape)
            d_fake = self.discrimitor(points_adv.detach())
            d_loss_fake = torch.mean(d_fake ** 2)
            pc = pc.transpose(2, 1)[:, :3, :].contiguous()

            d_real = self.discrimitor(pc)
            d_loss_real = torch.mean((d_real - 1) ** 2)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        if optimizer_idx == 1:
            points_adv = self.forward(pc, target_labelsg)
            d_fake = self.discrimitor(points_adv)
            g_loss = torch.mean((d_fake - 1) ** 2)
            pred, _ = self.attack_model(points_adv)
            pred_loss = self.attack_model_criterion(pred, target_labels)
            pc = pc.transpose(2, 1)[:, :3, :].contiguous()
            generator_loss = torch.mean((pc - points_adv) ** 2)
            g_loss = generator_loss + self.TAU * pred_loss + g_loss
            pred_choice = pred.data.max(1)[1]
            acc = pred_choice.eq(target_labels.data).float().sum()
            self.correct_adv += acc
            pred_original,_ = self.attack_model(pc)
            pred_choice_original = pred_original.data.max(1)[1]
            acc_original = pred_choice_original.eq(labels.data).float().sum()
            self.correct+=acc_original
            # print(acc_original)
            self.total+=labels.shape[0]
            tqdm_dict = {'g_loss': g_loss,
                         'attack sucess rate':100.*self.correct_adv/self.total,
                         'curadv':self.correct_adv,
                         'total':self.total,
                         'original cls rate':100.*self.correct/self.total,
                         }
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
    def on_epoch_end(self):
        self.correct = 0
        self.correct_adv = 0
        self.total = 0
    # def validation_step(self, batch, batch_idx):
    #     pc, labels = batch
    #     target_labels = self._generate_labels(labels)
    #     target_labels = target_labels.long().cuda()
    #
    #     points_adv = self.forward(pc, target_labels)
    #     pc = pc.permute(0, 2, 1).contiguous()[:, :3, :]
    #     d_fake = self.discrimitor(points_adv.detach())
    #     d_loss_fake = torch.mean(d_fake ** 2)
    #
    #     d_real = self.discrimitor(pc)
    #     d_loss_real = torch.mean((d_real - 1) ** 2)
    #     d_loss = 0.5 * (d_loss_real + d_loss_fake)
    #
    #     d_fake = self.discrimitor(points_adv)
    #     g_loss = torch.mean((d_fake - 1) ** 2)
    #     pred, _ = self.attack_model(points_adv)
    #     pred_loss = self.attack_model_criterion(pred, target_labels)
    #     generator_loss = torch.mean((pc - points_adv) ** 2)
    #     g_loss = generator_loss + self.TAU * pred_loss + g_loss
    #     pred_choice = pred.data.max(1)[1]
    #     acc = pred_choice.eq(target_labels.data).float().mean()
    #     tqdm_dict = {'g_loss': g_loss,'d_loss': d_loss,'attack sucess rate':acc}
    #     output = OrderedDict({
    #         'loss': g_loss,
    #         'progress_bar': tqdm_dict,
    #         'log': tqdm_dict
    #     })
    #     return output



    def configure_optimizers(self):
        lr1 = self.hparams['l1']
        lr2 = self.hparams['l2']
        b1 = self.hparams['b1']
        b2 = self.hparams['b2']
        b3 = self.hparams['b3']
        b4 = self.hparams['b4']
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr1, betas=(b3, b4))
        opt_d = torch.optim.Adam(self.discrimitor.parameters(), lr=lr2, betas=(b1, b2))
        return [opt_d, opt_g], []

    def prepare_data(self):
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                # d_utils.PointcloudRotate(),
                # d_utils.PointcloudRotatePerturbation(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
                # d_utils.PointcloudRandomInputDropout(),
            ]
        )

        self.train_dset = ModelNet40Cls(
            self.hparams["num_points"], transforms=train_transforms, train=True
        )
        self.val_dset = ModelNet40Cls(
            self.hparams["num_points"], transforms=None, train=False
        )

    def _build_dataloader(self, dset, mode):
        return DataLoader(
            dset,
            batch_size=self.hparams["batch_size"],
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")

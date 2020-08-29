import os
import csv
import argparse
import torch
from torch.nn import functional as F
from torch import nn
from pointnet2.utils import IOStream
import random
from torch.utils.data import Dataset, DataLoader
import sys
import h5py
from pointnet2.models.lgnet import Generator
from pointnet2.models.point_resnet import Discriminator
from pointnet2.models.pointnet import PointNetCls
import pointnet2.data.data_utils as d_utils
from pointnet2.data.ModelNet40Loader import ModelNet40Cls
from pointnet2.utils import progress_bar, adjust_lr_steep, log_row
from torchvision import transforms
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import plotly.graph_objs as go
from pointnet2.transforms_3d import *

########################################
## Set hypeparameters
########################################
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pointnet', help='choose model type')
parser.add_argument('--data', type=str, default='modelnet40', help='choose data set')
parser.add_argument('--seed', type=int, default=0, help='manual random seed')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--step', nargs='+', default=[50, 80, 120, 150],
                    help='epochs when to change lr, for example type "--adj_step 50 80 120 150" in command line')
parser.add_argument('--dr', nargs='+', default=[0.1, 0.1, 0.2, 0.2], help='decay rates of learning rate')
parser.add_argument('--resume', type=str, default='example1', help='resume path')
parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform")
parser.add_argument('--lambda_ft', type=float, default=0.001, help="lambda for feature transform")
parser.add_argument('--augment', type=int, default=1, help='data argment to increase robustness')
parser.add_argument('--name', type=str, default='train', help='name of the experiment')
parser.add_argument('--note', type=str, default='', help='notation of the experiment')
parser.add_argument('--adv_path', default='LGGAN', help='output adversarial example path [default: LGGAN]')
parser.add_argument('--lggan', type=str,default='lggan', help='run file store place')
parser.add_argument('--tau', type=float, default=1e2, help='balancing weight for loss function [default: 1e2]')
args = parser.parse_args()
args.adj_lr = {'steps': [int(temp) for temp in args.step],
               'decay_rates': [float(temp) for temp in args.dr]}
args.feature_transform, args.augment = bool(args.feature_transform), bool(args.augment)
### Set random seed
args.seed = args.seed if args.seed > 0 else random.randint(1, 10000)
if not os.path.exists('/home/wei/Pointnet2_PyTorch/pointnet2/models/checkpoints/lggan'):
    os.mkdir('/home/wei/Pointnet2_PyTorch/pointnet2/models/checkpoints/lggan')
io = IOStream('/home/wei/Pointnet2_PyTorch/pointnet2/models/checkpoints/lggan/run.log')
io.cprint(str(args))
TAU = args.tau
epochs = 100

# create adversarial example path
ADV_PATH = "/home/wei/Pointnet2_PyTorch/pointnet2/models/checkpoints/LGGAN"
if not os.path.exists('results'): os.mkdir('results')
ADV_PATH = os.path.join('results', ADV_PATH)
if not os.path.exists(ADV_PATH): os.mkdir(ADV_PATH)
ADV_PATH = os.path.join(ADV_PATH, 'test')
checkpoints_dir = "/home/wei/Pointnet2_PyTorch/LGGAN"

NUM_CLASSES = 40


def write_h5(data, data_orig, label, label_orig, num_batches):

    h5_filename = ADV_PATH+str(num_batches)+'.h5'
    h5f = h5py.File(h5_filename, 'w')
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('orig_data', data=data_orig)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('orig_label', data=label_orig)
    h5f.close()
def generate_labels(labels):
    targets = np.zeros(np.size(labels))
    for i in range(len(labels)):
        rand_v = random.randint(0, NUM_CLASSES-1)
        while labels[i]==rand_v:
            rand_v = random.randint(0, NUM_CLASSES-1)
        targets[i] = rand_v
    targets = targets.astype(np.int32)

    return targets

def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res
def draw(points):
    trace = go.Scatter3d(
        x=points[0, 0, :], y=points[0, 1, :], z=points[0, 2, :], mode='markers', marker=dict(
            size=12,
            color=points[0, 2, :],  # set color to an array/list of desired values
            colorscale='Viridis'
        )
    )
    layout = go.Layout(title='3D Scatter plot')
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


if __name__ == '__main__':
    g_lr = 1e-3
    d_lr= 1e-5
    train_transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout(),
        ]
    )
    test_tfs = None
    train_data = ModelNet40Cls(
            args.num_points, transforms=train_transforms, train=True
        )
    test_data = ModelNet40Cls(
            args.num_points, transforms=train_transforms, train=False
        )
    train_loader = DataLoader(train_data, num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, num_workers=4,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    ########################################
    ## Intiate model
    ########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_model = PointNetCls(NUM_CLASSES, args.feature_transform).to(device)
    attack_model=attack_model.to(device)
    print('=====> Loading from checkpoint...')
    checkpoint = torch.load('/home/wei/Pointnet2_PyTorch/pointnet2/models/checkpoints/%s.pth' % args.resume)
    args = checkpoint['args']

    torch.manual_seed(args.seed)
    print("Random Seed: ", args.seed)

    attack_model.load_state_dict(checkpoint['model_state_dict'])


    generator = Generator(up_ratio=1).to(device)
    discriminator = Discriminator(torch.nn.functional.leaky_relu,num_point=args.num_points).to(device)

    attack_model_criterion = F.cross_entropy
    generator_criterion = F.mse_loss
    d_optim = torch.optim.Adam(list(discriminator.parameters()),lr=d_lr,betas=(0.5, 0.999))
    g_optim = torch.optim.Adam(list(generator.parameters()),lr=g_lr,betas=(0.9, 0.999))
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=20, gamma=0.1)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, step_size=20, gamma=0.1)

    attack_model.eval()
    best_acc = 0.0
    best_epoch = 0
    if os.path.exists(str(checkpoints_dir) + '/best_model.pth'):
        print('=====> Loading from checkpoint...')
        checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth')
        best_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['g_model_state_dict'])
        discriminator.load_state_dict(checkpoint['d_model_state_dict'])
        best_acc = checkpoint['best_acc']
        d_optim.load_state_dict(checkpoint['d_optimizer_state_dict'])
        g_optim.load_state_dict(checkpoint['g_optimizer_state_dict'])
        print('Successfully resumed!')

    for epoch in range(epochs):

        print("epochs:",epoch)
        #### TRAIN #########
        correct = 0
        correct_adv = 0
        total = 0
        generator.train()
        discriminator.train()
        g_loss_total = 0.0
        d_loss_total = 0.0
        print("TRAIN")
        for i, data in enumerate(train_loader, 0):
            points, label = data
            # print(points.shape,label.shape)
            target_labels = generate_labels(label.numpy())
            target_labels = torch.tensor(target_labels,dtype=torch.long).to(device)
            points, label = points.to(device), label.to(device)
            # print(points.shape)
            # draw(points.cpu())
            # print(points.shape)
            g_optim.zero_grad()
            d_optim.zero_grad()
            labelg = one_hot(label,40).to(device)
            target_labelsg = one_hot(target_labels, 40).to(device)
            points_adv = generator(points, target_labelsg)
            # print(points_adv.shape)

            d_fake = discriminator(points_adv.detach())
            d_loss_fake = torch.mean(d_fake ** 2)
            d_loss_fake.backward()
            points = points.transpose(2, 1)[:,:3,:]

            d_real = discriminator(points)
            d_loss_real = torch.mean((d_real - 1) ** 2)
            d_loss_real.backward()
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_optim.step()

            d_fake = discriminator(points_adv)
            g_loss = torch.mean((d_fake - 1) ** 2)
            g_loss.backward(retain_graph=True)
            pred, _ = attack_model(points_adv)
            pred_choice = pred.data.max(1)[1]
            correct_adv += pred_choice.eq(target_labels.data).cpu().sum()
            total += label.size(0)
            pred_loss = attack_model_criterion(pred, target_labels)
            generator_loss = torch.mean((points-points_adv)**2)
            g_loss = generator_loss + TAU * pred_loss + g_loss

            g_loss.backward()
            g_optim.step()

            g_loss_total+=g_loss.item()
            d_loss_total+=d_loss.item()
            progress_bar(i, len(train_loader),
                         'Generator Loss: %.3f | Discrimitor Loss: %.3f| Attack Success rate: %.3f%% (%d/%d)'
                         % (g_loss_total / total, d_loss_total / total,
                            100. * correct_adv / total, correct_adv, total))
        g_scheduler.step()
        d_scheduler.step()
        #### TEST ######
        correct = 0
        correct_adv = 0
        total = 0
        generator.eval()
        discriminator.eval()
        g_loss_total = 0.0
        d_loss_total = 0.0
        print("TEST")
        # with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            points, label = data
            target_labels = generate_labels(label.numpy())
            target_labels = torch.tensor(target_labels, dtype=torch.long).to(device)
            points, label = points.to(device), label.to(device)

            labelg = one_hot(label, 40).to(device)
            target_labelsg = one_hot(target_labels, 40).to(device)
            points_adv = generator(points, target_labelsg)

            d_fake = discriminator(points_adv.detach())
            d_loss_fake = torch.mean(d_fake ** 2)
            points = points.transpose(2, 1)[:, :3, :]

            d_real = discriminator(points)
            d_loss_real = torch.mean((d_real - 1) ** 2)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_fake = discriminator(points_adv)
            g_loss = torch.mean((d_fake - 1) ** 2)
            pred, _ = attack_model(points_adv)
            pred_choice = pred.data.max(1)[1]
            correct_adv += pred_choice.eq(target_labels.data).cpu().sum()
            total += label.size(0)
            pred_loss = attack_model_criterion(pred, target_labels)
            generator_loss = torch.mean((points - points_adv) ** 2)
            g_loss = generator_loss + TAU * pred_loss + g_loss


            g_loss_total+=g_loss.item()
            d_loss_total+=d_loss.item()
            progress_bar(i, len(test_loader),
                         'Generator Loss: %.3f | Discrimitor Loss: %.3f| Attack Success rate: %.3f%% (%d/%d)'
                         % (g_loss_total / total, d_loss_total/ total,
                          100. * correct_adv/ total, correct_adv, total))
        acc = 100.*correct_adv/total
        if acc>=best_acc:
            best_acc=acc
            best_epoch = epoch+1
            print("better model find")
            savepath = str(checkpoints_dir) + '/best_model.pth'
            state = {
                'epoch': best_epoch,
                'best_acc': best_acc,
                'g_model_state_dict': generator.state_dict(),
                'd_model_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict':g_optim.state_dict(),
                'd_optimizer_state_dict':d_optim.state_dict(),
                'g_steplr':g_scheduler.state_dict(),
                'd_steplr':d_scheduler.state_dict(),
            }
            torch.save(state, savepath)


import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pointnet2.models.pointnet import PointNetCls
from torch.utils.data import DataLoader, DistributedSampler
import pointnet2.data.data_utils as d_utils
from pointnet2.data.ModelNet40Loader import ModelNet40Cls
from torchvision import transforms
import numpy as np
import time
import sys

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    # Reset for new bar
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    #L.append('  Step: %s' % format_time(step_time))
    L.append(' Time: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

@hydra.main("config/config.yaml")
def main(cfg):
    '''

    Parameters
    ----------
    cfg:{'gpus': [0], 'optimizer': {'weight_decay': 0.0, 'lr': 0.001, 'lr_decay': 0.7, 'bn_momentum': 0.5,
    'bnm_decay': 0.5, 'decay_step': 20000.0}, 'task_model': {'class': 'pointnet2.models.PointNet2ClassificationSSG',
    'name': 'cls-ssg'}, 'model': {'use_xyz': True}, 'distrib_backend': 'dp', 'num_points': 1024, 'epochs': 200,
    'batch_size': 32}
    cfg.task_model:{'class': 'pointnet2.models.PointNet2ClassificationSSG', 'name': 'cls-ssg'}
    hydra_params_to_dotdict(cfg):{'optimizer.weight_decay': 0.0, 'optimizer.lr': 0.001, 'optimizer.lr_decay': 0.7,
    'optimizer.bn_momentum': 0.5, 'optimizer.bnm_decay': 0.5, 'optimizer.decay_step': 20000.0, 'task_model.class':
     'pointnet2.models.PointNet2ClassificationSSG', 'task_model.name': 'cls-ssg', 'model.use_xyz': True, 'distrib_backend':
     'dp', 'num_points': 1024, 'epochs': 200, 'batch_size': 32}

    Returns
    -------

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model =PointNetCls(40,bool(1)).to(device)
    checkpoint = torch.load('/home/wei/Pointnet2_PyTorch/pointnet2/models/checkpoints/example1.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    # early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     monitor="val_acc",
    #     mode="max",
    #     save_top_k=2,
    #     filepath=os.path.join(
    #         cfg.task_model.name, "{epoch}-{val_loss:.2f}-{val_acc:.3f}"
    #     ),
    #     verbose=True,
    # )

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
    dset = ModelNet40Cls(
            2048, transforms=train_transforms, train=True
        )

    test_loader = DataLoader(
        dset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    model.eval()
    correct = 0
    total = 0
    class_acc = np.zeros((40, 3))
    for j, data in enumerate(test_loader, 0):
        points, label = data
        points, label = points.to(device), label.to(device)


        points = points.transpose(2, 1)  # to be shape batch_size*3*N
        points = points[:,:3,:]
        pred, trans_feat = model(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(label.cpu()):
            # print(pred_choice[target==cat].long().data)
            classacc = pred_choice[label == cat].eq(label[label == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[label == cat].size()[0])
            class_acc[cat, 1] += 1
        # print(pred.shape, label.shape)

        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(label.data).cpu().sum()
        total += label.size(0)
        progress_bar(j, len(test_loader), ' Test Acc: %.3f%% (%d/%d)'
                     % ( 100. * correct.item() / total, correct, total))



if __name__ == "__main__":
    main()

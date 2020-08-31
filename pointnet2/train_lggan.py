import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pointnet2.models.lggan_single import GAN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


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
    model =GAN({'l1':1e-3,'l2':1e-5,'b1':0.5,'b2':0.599,'b3':0.9,'b4':0.999,'batch_size':8,"num_points":2048})

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
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=200,
        # early_stop_callback=early_stop_callback,
        # checkpoint_callback=checkpoint_callback,
        # distributed_backend=cfg.distrib_backend,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()

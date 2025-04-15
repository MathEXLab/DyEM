import os
from omegaconf import OmegaConf
from argparse import ArgumentParser
import datetime
import torch
from lightning import Trainer, seed_everything
import time
import yaml
import numpy as np

from src.exp.exp_basic import model_dict, exp_dict, datamodule_dict
from src.utils.utils_lightning import get_callbacks, get_logger



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/dynamical_embedding/configs/ks/gcn/ks_gcn_init3_lead1.yaml")
    parser.add_argument("--seed", type=str, default=42)
    parser.add_argument('--device', nargs="*", type=int, default=[0])
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.general.output_dir = "{}_{}_{}_{}".format(
        cfg.general.task_name,
        cfg.data.data_name,
        cfg.model.model_name,
        cfg.general.cust_name,
    )

    if args.seed is not None:
        cfg.general.seed = args.seed
    if args.device is not None:
        cfg.train.devices = args.device
    seed_everything(cfg.general.seed, workers=True)

    #datamodule
    datamodule = datamodule_dict[cfg.general.task_name](cfg.data)

    # optimizer & scheduler
    optimizer = getattr(torch.optim, cfg.train.optimizer)
    print('Use optimizer:', cfg.train.optimizer)
    scheduler=None  # TODO

    # model
    model = exp_dict[cfg.general.task_name](
        net=model_dict[cfg.model.model_name].Model(cfg.model),
        optimizer=optimizer,
        scheduler=scheduler,
        compile=False,
        criterion=cfg.train.criterion,
        **cfg.train.hparams
    )

    # trainer
    callbacks = get_callbacks(cfg.train.callbacks)
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        min_epochs=cfg.train.min_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        deterministic=True,
        default_root_dir=cfg.general.output_dir,
        callbacks=list(callbacks.values()),
        logger=get_logger(cfg.general.output_dir),
    )

    # train
    st = time.time()
    trainer.fit(model, datamodule=datamodule)
    print("Training time: ", time.time()-st)

    # test
    trainer.test(model, datamodule=datamodule, ckpt_path='best')

    # predict
    if hasattr(cfg.general, 'predict'):
        if cfg.general.predict== False:
            pass
    else:
        predictions = trainer.predict(model, datamodule=datamodule, ckpt_path='best') # for saving results
        pred = []
        true = []
        for i in range(len(predictions)):
            y_hat, y = predictions[i]
            pred.append(y_hat.detach().cpu().numpy())
            true.append(y.detach().cpu().numpy())

        # cocatenate all the predictions and true values in axis 0
        pred = np.concatenate(pred, axis=0).squeeze()
        true = np.concatenate(true, axis=0).squeeze()
        
        # save results
        # torch.save(predictions, os.path.join(os.path.dirname(callbacks['model_checkpoint'].best_model_path), "predictions.pt"))
        np.save(os.path.join(os.path.dirname(callbacks['model_checkpoint'].best_model_path), "predictions.npy"), pred)
        np.save(os.path.join(os.path.dirname(callbacks['model_checkpoint'].best_model_path), "true.npy"), true)

        # save config
        with open(args.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg_dict['general']['seed'] = cfg.general.seed # override seed

        with open(os.path.join(os.path.dirname(callbacks['model_checkpoint'].best_model_path), "config.yaml"), 'w') as f:
            yaml.dump(cfg_dict, f)
        





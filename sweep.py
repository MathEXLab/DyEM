from omegaconf import OmegaConf, ListConfig
from argparse import ArgumentParser
import optuna
import torch
import time
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import CSVLogger
from lightning import Trainer, seed_everything

# from data_provider.data_factory import data_provider
from src.exp.exp_basic import model_dict, exp_dict, datamodule_dict
from src.utils.utils_lightning import get_callbacks, get_logger




SAMPLER = {
    'TPESampler': optuna.samplers.TPESampler,
}


if __name__ == "__main__":
    parser = ArgumentParser(description='sweep')
    parser.add_argument('--config', type=str, default= '/home/dynamical_embedding/configs/lorenz/lstm/direct/lorenz_BiLSTM_init3_lead1_tune.yaml')
    parser.add_argument('--sweep_config', type=str, default= '/home/dynamical_embedding/configs/lorenz/sweep/lstm/lorenz_BiLSTM_init3_lead1_sweep.yaml')
    parser.add_argument('--device', nargs="*", type=int, default= [0]) 
    args = parser.parse_args()

    # process config
    cfg = OmegaConf.load(args.config) 
    cfg.general.output_dir = "{}_{}_{}_{}".format(
        cfg.general.task_name,
        cfg.data.data_name,
        cfg.model.model_name,
        cfg.general.cust_name,
    )

    sweep_cfg = OmegaConf.load(args.sweep_config)

    seed_everything(cfg.general.seed, workers=True)

    
    def objective(trial: optuna.trial.Trial) -> float:
    
        hparams_dict = {}
        # we tune model, data(batch), and train
        # model
        if hasattr(sweep_cfg, 'model'):
            for key, value in sweep_cfg.model.items():
                cfg.model[key] = trial.suggest_categorical(f'{key}', value)
                hparams_dict[key] = cfg.model[key]
        # data
        if hasattr(sweep_cfg, 'data'):
            for key, value in sweep_cfg.data.items():
                cfg.data[key] = trial.suggest_categorical(f'{key}', value)
                hparams_dict[key] = cfg.data[key]
        # train
        if hasattr(sweep_cfg, 'train'):
            for key, value in sweep_cfg.train.hparams.items():
                cfg.train.hparams[key] = trial.suggest_categorical(f'{key}', value)
                hparams_dict[key] = cfg.train.hparams[key]
        # for key, value in sweep_cfg.model.items():
        #     cfg.model[key] = trial.suggest_categorical(f'{key}', value)
        #     hparams_dict[key] = cfg.model[key]
        print(f'Current hparams: {hparams_dict}')

        # instantiating datamodule
        datamodule = datamodule_dict[cfg.general.task_name](cfg.data)
    
        optimizer = getattr(torch.optim, cfg.train.optimizer)

        # model
        model = exp_dict[cfg.general.task_name](
            net=model_dict[cfg.model.model_name].Model(cfg.model),
            optimizer=optimizer,
            scheduler=None,
            compile=False,
            **cfg.train.hparams
        )

        # train
        callbacks = get_callbacks(cfg.train.callbacks)
        trainer = Trainer(
            max_epochs=cfg.train.max_epochs,
            min_epochs=cfg.train.min_epochs,
            accelerator=cfg.train.accelerator,
            devices=args.device,
            deterministic=True,
            default_root_dir=cfg.general.output_dir,
            callbacks=list(callbacks.values()),
            logger=get_logger(cfg.general.output_dir),
        )

        trainer.logger.log_hyperparams(hparams_dict)
        st = time.time()
        trainer.fit(model, datamodule=datamodule)
        print("Training time: ", time.time()-st)

        return trainer.callback_metrics["val_loss"].item()
    
    # optuna setting
    sampler = SAMPLER[sweep_cfg.sampler.sampler_name](seed = cfg.general.seed, n_startup_trials=sweep_cfg.sampler.n_startup_trials)
    study_name = sweep_cfg.study_name
    study = optuna.create_study(direction=sweep_cfg.direction,
                                sampler=sampler,
                                study_name=study_name,
                                storage=f"sqlite:///{cfg.general.output_dir}_sweep.db",
                                load_if_exists=True,)
    study.optimize(objective, n_trials=sweep_cfg.n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Value: {}".format(trial.value))

    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
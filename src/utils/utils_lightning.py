# To Do List

# [] unbatched predictions for calculating metrics

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger


def get_callbacks(configs):

    model_checkpoint = ModelCheckpoint(
        monitor=configs.monitor,
        mode=configs.mode,
        save_top_k=3,
        filename='epoch{epoch:03d}_val_loss{val_loss:.2f}',
        auto_insert_metric_name=False,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=configs.monitor,
        patience=configs.patience,
        mode=configs.mode,
        verbose=True,
    )

    return {'model_checkpoint':model_checkpoint, 
            'early_stopping':early_stopping}


def get_logger(configs):
    return CSVLogger("logs", name=configs)

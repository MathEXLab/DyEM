from src.model.models import BiLSTM, Transformer, CNN1d, DeepONet,MLP,FNO,DDPM1d,Vit, CNN2d, DeepONet2d, FNO2d,TCN, GCN
from src.datamodule.datamodule_regression import RegLitDataModule, WeightedDataModule
from .exp_classification import ClsLitModel
from .exp_regression import RegLitModel, DyEmbRegLitModel, BaseLitModule


exp_dict = {
    'classification': ClsLitModel,
    'regression': RegLitModel,
    'dyemb_regression': DyEmbRegLitModel,
    'weighted_regression': BaseLitModule,
}

datamodule_dict = {
    # 'classification': ClassificationLitDataModule,
    'regression': RegLitDataModule,
    'dyemb_regression': RegLitDataModule,
    'weighted_regression': WeightedDataModule,
}

model_dict = {
    'Transformer': Transformer,
    'BiLSTM': BiLSTM,
    'CNN1d': CNN1d,
    'deeponet': DeepONet,
    'MLP': MLP,
    'DDPM1d': DDPM1d, # autoencoder-like, not for regression
    'FNO': FNO,
    'Vit': Vit,
    'CNN2d': CNN2d,
    'deeponet2d': DeepONet2d,
    'FNO2d': FNO2d,
    'TCN': TCN,
    'GCN': GCN,
}



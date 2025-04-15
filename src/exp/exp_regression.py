import torch
import torch.nn as nn
import lightning as L
from typing import Any, Dict, Tuple
from torchmetrics import MeanMetric
import numpy as np

# from ..utils.custom_criterion import *
# from .. import pypardi.di_evaluate as di_eval
# import src.pypardi.di_evaluate as di_eval
from src.utils.scaling.softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from src.utils.dyemb_losses import DyEmb_MSELoss, D_MSELoss, DI_WMSELoss

class Weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return torch.mean(torch.matmul(weights,(inputs - targets)**2)) 
    
class Normalized_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self,inputs,targets):
        return self.mse(inputs,targets) / torch.mean(targets**2)

# TODO add DE_MSELoss
# class DyEmb_MSELoss(nn.Module):
#     def __init__(self, base_attractor, **kwargs):
#         super().__init__()
#         # calculate di on cpu
#         self.device = 'cpu'
#         self.data = base_attractor
#         self.mse = nn.MSELoss()
    
#     # calculate di with respective to this prediction
#     def cal_di(self, data, new):
#         new = new.to(self.device).detach().numpy()
#         data = data.reshape(data.shape[0],-1,1)
#         new = new.reshape(new.shape[0],-1,1)
#         di_temp = []
#         theta_temp = []
#         # st = time.time()
#         for i in range(new.shape[0]):
#             temp = di_eval.compute(
#                 data, new[i][None,...], ql=0.98, p=2, theta_fit="sueveges",
#                 p_value=None, dql=None, exp_test='anderson',
#                 p_cross=None, distributed='none',comm=None
#             )
            
#             di_temp.append(temp['d'])
#             theta_temp.append(temp['theta'])
#             # concat X and new[i] to X
#             # X = np.concatenate((X, new[i][None,...]), axis=0) # we use same X (training data) for all new[i]
#         # total_time = time.time()-st
#         # time_per_sample = total_time/new.shape[0]

#         di = np.array(di_temp)
#         theta = np.array(theta_temp)

#         # to tensor
#         di = torch.tensor(di).float().squeeze()
#         theta = torch.tensor(theta).float().squeeze()
        
#         return di, theta
    
#     def p_norm(self,y,y_hat, p=2):
#         norm = torch.linalg.vector_norm(y_hat-y, ord=p, dim = None) # 1d tensor
#         return norm
    
#     def scaling(self, d_loss, theta_loss):
#         weight_d = 1
#         weight_theta = 1
#         return weight_d*d_loss + weight_theta*theta_loss

#     def forward(self,inputs, targets, **kwargs):
#         self.map_device = inputs.device
#         # TODO use pre-calculated di instead of calculating di for true
#         if 'd' in kwargs and 'theta' in kwargs:
#             d_true = kwargs['d']
#             theta_true = kwargs['theta']
#         else:
#             d_true, theta_true = self.cal_di(self.data, targets)    # shape (batch_size,), (batch_size,)
#         d_pred, theta_pred = self.cal_di(self.data, inputs)    # shape (batch_size,), (batch_size,)
#         d_loss = self.p_norm(d_true, d_pred)
#         theta_loss = self.p_norm(theta_true, theta_pred)
#         # dy_loss = self.scaling(norm_did_d, norm_did_theta) 
#         mse_loss = self.mse(inputs, targets)
#         d_loss.requires_grad= True
#         theta_loss.requires_grad= True
#         loss = self.scaling(d_loss, theta_loss) + mse_loss  
#         loss = loss.to(self.map_device)
#         # loss.backward()

#         return loss, d_loss,theta_loss, mse_loss

    
class RegLitModel(L.LightningModule):

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        lr: float,
        criterion: str = 'mse', # choose loss, e.g 'mse', 'l1'
        **kwargs,
    ) -> None:
        """I

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether compile model for fast training, use with cautious.
        """
        super().__init__()

        if criterion.lower() == "mse":
            self.criterion = torch.nn.MSELoss()
        elif criterion.lower() == "wmse":
            assert hasattr(self, 'weights'), 'weights should be provided'
            self.criterion = Weighted_MSELoss()
        elif criterion.lower() == "nmse":
            self.criterion = Normalized_MSELoss()
        elif criterion.lower() == "di_wmse":
            d_base = kwargs.get('d_base', None)
            theta_base = kwargs.get('theta_base', None)
            assert d_base is not None and theta_base is not None, 'd_base and theta_base should be provided'
            self.d_base = np.load(d_base)
            self.theta_base = np.load(theta_base)
            self.d_base = torch.tensor(self.d_base).float()
            self.theta_base = torch.tensor(self.theta_base).float()
            self.criterion = DI_WMSELoss(d_base=self.d_base, theta_base=self.theta_base)
        else:
            raise ValueError(f"Criterion {criterion} not supported.")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # special for ddpm
        self.return_loss = kwargs.get('return_loss',False)    # wether model return predict value or return loss

        # default value for optimizer
        self.opt_params = kwargs.get('opt_params', None)

        # default reg_steps
        self.reg_steps = int(kwargs.get('multistep_loss', 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model."""

        output = self.model(x)
        return output
    
    def ar_step(self, x, batch_idx: int) -> torch.Tensor:
        y_hats = []
        for i in range(self.reg_steps):
            y_hat = self.model(x)
            y_hats.append(y_hat)
            # concatenate y_hat to x, and remove the first element
            x = torch.cat((x, y_hat.unsqueeze(1)), dim=1)[:,1:,:]
        y_hats = torch.stack(y_hats)
        y_hats = y_hats.permute(1,0,2)
        return y_hats
    
    def on_train_epoch_start(self):
        """Reset metrics at the start of each training epoch."""
        self.train_loss.reset()
        self.val_loss.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of inputs and target
            outputs.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, y = batch
        
        if self.reg_steps > 1:
            y_hat = self.ar_step(x,batch_idx)
        else:
            y_hat = self.model(x)

        if self.return_loss:
            loss = y_hat
        else:
            loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.train_loss(loss)

        # log metrics
        self.log("train_loss", self.train_loss.compute(), on_step=False, on_epoch=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr,)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """_summary_

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): _description_
            batch_idx (int): _description_
        """
        x, y = batch

        if self.reg_steps > 1:
            y_hat = self.ar_step(x,batch_idx)
        else:
            y_hat = self.model(x)

        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.val_loss(loss)

        # update and log metrics
        self.log("val_loss", self.val_loss.compute(),on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        '''Reset metrics at the start of each test epoch.'''
        self.test_loss.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, y = batch
        y_hat = self.model(x)
        if self.reg_steps > 1:
            y = y[:,0,:]    # only save the first step
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.test_loss(loss)

        # update and log metrics
        self.log("test_loss", self.test_loss.compute(), on_step=False, on_epoch=True)
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        if self.reg_steps > 1:
            y = y[:,0,:]    # only save the first step
        y_hat = self.model(x)
        return y_hat, y

    def setup(self, stage: str) -> None:
        """
        This hook is called on every process when using DDP.
        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.opt_params is not None:
            optimizer = self.optimizer(params=self.parameters(), lr = self.lr, **self.opt_params)
            print('Using optimizer with params:', self.opt_params)
        else:
            optimizer = self.optimizer(params=self.parameters(), lr = self.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    

# TODO: re-organize code
class DyEmbRegLitModel(L.LightningModule):

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        lr: float,
        base_attractor: np.ndarray,
        criterion: str = 'demse', # choose loss, e.g 'mse', 'l1'
        **kwargs,
    ) -> None:
        """
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether compile model for fast training, use with cautious.
        """
        super().__init__()

        self.base_attractor = base_attractor

        if criterion.lower() == "demse":
            self.mse = torch.nn.MSELoss()
            self.dy_loss = DyEmb_MSELoss(base_attractor)
        elif criterion.lower() == "dmse":
            self.mse = torch.nn.MSELoss()
            self.dy_loss = D_MSELoss(base_attractor)
        else:
            raise ValueError(f"Criterion {criterion} not supported.")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr

        self.train_loss = MeanMetric()
        self.train_d_loss = MeanMetric()
        self.train_theta_loss = MeanMetric()
        self.train_mse_loss = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_d_loss = MeanMetric()
        self.val_theta_loss = MeanMetric()
        self.val_mse_loss = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_d_loss = MeanMetric()
        self.test_theta_loss = MeanMetric()
        self.test_mse_loss = MeanMetric()

        # parameters for scaling
        self.scaling = self.hparams.scaling
        self.weights =torch.tensor([1,1,1]) # initial weights
        if self.scaling == 'softadapt':
            print('Using SoftAdapt scaling')
            assert self.hparams.beta is not None, 'beta should be provided'
            self.softadapt_object = SoftAdapt(beta=self.hparams.beta)
        elif self.scaling == 'lossweighted':
            print('Using LossWeightedSoftAdapt scaling')
            assert self.hparams.beta is not None, 'beta should be provided'
            self.softadapt_object = LossWeightedSoftAdapt(beta=self.hparams.beta)
        elif self.scaling == 'normalized':
            print('Using NormalizedSoftAdapt scaling')
            assert self.hparams.beta is not None, 'beta should be provided'
            self.softadapt_object = NormalizedSoftAdapt(beta=self.hparams.beta)
        elif self.scaling == 'none':
            self.softadapt_object = None
            print('No scaling, use uniform weights')

        # flag for use_precomputed_di
        if kwargs.get('use_precomputed_di', False): # whether to use precomputed di
            self.use_precomputed_di = True
        else:
            self.use_precomputed_di = False

        # flag for interval of dy embedding forcing
        self.dy_interval = kwargs.get('dy_interval', 1) # interval of updating dy embedding
        self.dy_embed_this_epoch = False # used to control the interval of updating dy embedding

        # whether to parallelize di calculation
        self.parallel_cores = kwargs.get('parallel_cores', 1)   # number of cores to parallelize di calculation， if 1, calculate loss each step not epoch

        # default value for optimizer
        self.opt_params = kwargs.get('opt_params', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model."""
        output = self.model(x)
        return output
    
    def on_train_epoch_start(self):
        """Reset metrics at the start of each training epoch."""
        # whether to update dy dy embedding
        if self.current_epoch % self.dy_interval == 0:
            self.dy_embed_this_epoch = True
        else:
            self.dy_embed_this_epoch = False

        # mse loss are calculated in all conditions
        self.train_mse_losses = []
        self.train_loss.reset()
        self.train_mse_loss.reset()

        if self.dy_embed_this_epoch:
            self.train_pred = []    
            self.train_d_losses = []
            self.train_theta_losses = []
            self.train_d_loss.reset()  
            self.train_theta_loss.reset()
            if self.use_precomputed_di:
                self.train_d = []  # concatenate d, theta from input data
                self.train_theta = []

        # update weights
        # TODO: update weights at the begin of each epoch?
        # if self.dy_embed_this_epoch:


        # TODO: move to cal epoch start
        # self.val_loss.reset()
        # self.val_d_loss.reset()  
        # self.val_theta_loss.reset()
        # self.val_mse_loss.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of inputs and target
            outputs.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if self.use_precomputed_di: # whether to use precomputed di
            x, y, d, theta = batch
        else:
            x, y = batch
        
        y_hat = self.model(x)
        self.train_mse_losses.append(self.mse(y_hat.squeeze(), y.squeeze()))

        # need to judge: 1) whether use di as forcing in this epoch; 2) whether use precomputed di
        # append mse error, d and theta pred for di calculation each epoch
        # append mse error

        if self.dy_embed_this_epoch:
            # TODO gather d, theta, pred
            pass
            # self.train_pred.append(y_hat)
            # if self.use_precomputed_di:
            #     self.train_d.append(d)
            #     self.train_theta.append(theta)

        if self.parallel_cores == 1:
            # calculate loss in each step
            if self.use_precomputed_di:
                # TODO
                # loss, d_loss,theta_loss, mse_loss = self.dy_loss(y_hat.squeeze(), y.squeeze(), d=d, theta=theta)
                pass
            else:
                loss = self.mse(y_hat.squeeze(), y.squeeze())
                if self.dy_embed_this_epoch:
                    loss, d_loss,theta_loss, mse_loss = self.dy_loss(y_hat.squeeze(), y.squeeze())
                    loss = self.weights[0] * d_loss + self.weights[1] * theta_loss + self.weights[2] * mse_loss # weights are updated at the end of epoch
                    self.train_d_loss(d_loss)
                    self.train_theta_loss(theta_loss)
                    self.train_mse_loss(mse_loss)
                    self.train_d_losses.append(d_loss)
                    self.train_theta_losses.append(theta_loss)
                    self.log("train_d_loss", self.train_d_loss.compute(), on_step=False, on_epoch=True)
                    self.log("train_theta_loss", self.train_theta_loss.compute(), on_step=False, on_epoch=True)
                    self.log("train_mse_loss", self.train_mse_loss.compute(), on_step=False, on_epoch=True)

            self.train_loss(loss)
            self.log("train_loss", self.train_loss.compute(), on_step=False, on_epoch=True)

        else:
            # TODO MPI
            pass

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr,)

        return loss
    
    # TODO: use mpi to reduce traing time 
    def on_train_epoch_end(self):
        '''Concatenate all losses in an epoch'''
        # self.train_loss(torch.stack(self.train_losses).mean())
        # self.train_d_loss(torch.stack(self.train_d_losses).mean())
        # self.train_theta_loss(torch.stack(self.train_theta_losses).mean())
        # self.train_mse_loss(torch.stack(self.train_mse_losses).mean())
        # every 10 epochs, calculate di
        
        if self.dy_embed_this_epoch:
            # update weights
            if self.softadapt_object is not None:
                self.weights = self.softadapt_object.get_component_weights(torch.stack(self.train_d_losses), torch.stack(self.train_theta_losses), torch.stack(self.train_mse_losses))
            else:
                self.weights = torch.tensor([1,1,1])

            # log weights
            self.log("weight_d", self.weights[0], on_step=False, on_epoch=True)
            self.log("weight_theta", self.weights[1], on_step=False, on_epoch=True)
            self.log("weight_mse", self.weights[2], on_step=False, on_epoch=True)

    #     # log metrics
    #     self.log("train_loss", self.train_loss, on_step=False, on_epoch=True)
    #     self.log("train_d_loss", self.train_d_loss, on_step=False, on_epoch=True)
    #     self.log("train_theta_loss", self.train_theta_loss, on_step=False, on_epoch=True)
    #     self.log("train_mse_loss", self.train_mse_loss, on_step=False, on_epoch=True)
    
    def on_validation_epoch_start(self):
        '''Reset metrics at the start of each validation epoch.'''
        self.val_loss.reset()
        self.val_d_loss.reset()
        self.val_theta_loss.reset()
        self.val_mse_loss.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """_summary_

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): _description_
            batch_idx (int): _description_
        """
        if self.use_precomputed_di:
            x, y, d, theta = batch
        else:
            x, y = batch
        y_hat = self.model(x)
        
        if self.dy_embed_this_epoch:
            if self.use_precomputed_di:
                loss, d_loss,theta_loss, mse_loss = self.dy_loss(y_hat.squeeze(), y.squeeze(), d=d, theta=theta)
            else:
                loss, d_loss,theta_loss, mse_loss = self.dy_loss(y_hat.squeeze(), y.squeeze())

            self.val_loss(loss)
            self.val_d_loss(d_loss)
            self.val_theta_loss(theta_loss)
            self.val_mse_loss(mse_loss)

            # update and log metrics
            self.log("val_loss", self.val_loss.compute(),on_step=False, on_epoch=True)
            self.log("val_d_loss", self.val_d_loss.compute(),on_step=False, on_epoch=True)
            self.log("val_theta_loss", self.val_theta_loss.compute(),on_step=False, on_epoch=True)
            self.log("val_mse_loss", self.val_mse_loss.compute(),on_step=False, on_epoch=True)

        else:
            loss = self.mse(y_hat.squeeze(), y.squeeze())
            self.val_loss(loss)
            # self.val_mse_loss(loss)
            self.log("val_loss", self.val_loss.compute(),on_step=False, on_epoch=True)
            # self.log("val_mse_loss", self.val_mse_loss.compute(),on_step=False, on_epoch=True)

        # log metrics
        # self.log("val_loss", self.val_loss.compute(),on_step=False, on_epoch=True)
        # self.log("val_mse_loss", self.val_mse_loss.compute(),on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        '''Reset metrics at the start of each test epoch.'''
        self.test_loss.reset()
        self.test_d_loss.reset()
        self.test_theta_loss.reset()
        self.test_mse_loss.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # compute dy loss in test step, if use demse
        if self.use_precomputed_di:
            x, y, d, theta = batch
        else:
            x, y = batch
        y_hat = self.model(x)

        if self.use_precomputed_di:
            loss, d_loss,theta_loss, mse_loss = self.dy_loss(y_hat.squeeze(), y.squeeze(), d=d, theta=theta)
        else:
            loss, d_loss,theta_loss, mse_loss = self.dy_loss(y_hat.squeeze(), y.squeeze())
        # mse_loss = self.mse(y_hat.squeeze(), y.squeeze())

        loss = self.weights[0] * d_loss + self.weights[1] * theta_loss + self.weights[2] * mse_loss
        self.test_loss(loss)
        self.test_d_loss(d_loss)
        self.test_theta_loss(theta_loss)
        self.test_mse_loss(mse_loss)

        # update and log metrics
        self.log("test_loss", self.test_loss.compute(), on_step=False, on_epoch=True)
        self.log("test_d_loss", self.test_d_loss.compute(), on_step=False, on_epoch=True)
        self.log("test_theta_loss", self.test_theta_loss.compute(), on_step=False, on_epoch=True)
        self.log("test_mse_loss", self.test_mse_loss.compute(), on_step=False, on_epoch=True)
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if self.use_precomputed_di:
            x, y, d, theta = batch
        else:
            x, y = batch
        y_hat = self.model(x)
        return y_hat, y

    def setup(self, stage: str) -> None:
        """
        This hook is called on every process when using DDP.
        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.opt_params is not None:
            optimizer = self.optimizer(params=self.parameters(), lr = self.lr, **self.opt_params)
            print('Using optimizer with params:', self.opt_params)
        else:
            optimizer = self.optimizer(params=self.parameters(), lr = self.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class BaseLitModule(L.LightningModule):
    # this is the base module for later inherent
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        lr: float,
        criterion: str = 'mse', # choose loss, e.g 'mse', 'l1'
        **kwargs,
    ) -> None:
        super().__init__()

        if criterion.lower() == "mse":
            self.train_criterion = torch.nn.MSELoss()
            self.criterion = torch.nn.MSELoss()
        elif criterion.lower() == "wmse":
            self.train_criterion = Weighted_MSELoss()
            self.criterion = torch.nn.MSELoss()
            # weight can be computed or provided

        else:
            raise ValueError(f"Criterion {criterion} not supported.")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # default value for optimizer
        self.opt_params = kwargs.get('opt_params', None)

        # default reg_steps
        self.reg_steps = kwargs.get('multistep_loss', 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model."""

        output = self.model(x)
        return output
    
    def on_train_epoch_start(self):
        """Reset metrics at the start of each training epoch."""
        self.train_loss.reset()
        self.val_loss.reset()

    def ar_step(self, x, batch_idx: int) -> torch.Tensor:
        y_hats = []
        for i in range(self.reg_steps):
            y_hat = self.model(x)
            y_hats.append(y_hat)
            # concatenate y_hat to x, and remove the first element
            x = torch.cat((x, y_hat), dim=1)[:,1:]
        y_hats = torch.stack(y_hats)
        return y_hats

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of inputs and target
            outputs.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, y, weight = batch

        if self.reg_steps > 1:
            y_hat = self.ar_step(x,batch_idx)
        else:
            y_hat = self.model(x)

        if len(batch) == 2:
            loss = self.train_criterion(y_hat.squeeze(), y.squeeze())
        elif len(batch) == 3:
            loss = self.train_criterion(y_hat.squeeze(), y.squeeze(), weight.squeeze())

        self.train_loss(loss)

        # log metrics
        self.log("train_loss", self.train_loss.compute(), on_step=False, on_epoch=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr,)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """_summary_

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): _description_
            batch_idx (int): _description_
        """
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, y, weight = batch

        y_hat = self.model(x)

        if len(batch) == 2:
            loss = self.train_criterion(y_hat.squeeze(), y.squeeze())
        elif len(batch) == 3:
            loss = self.train_criterion(y_hat.squeeze(), y.squeeze(), weight.squeeze())

        # loss = self.criterion(y_hat.squeeze(), y.squeeze())


        self.val_loss(loss)

        # update and log metrics
        self.log("val_loss", self.val_loss.compute(),on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        '''Reset metrics at the start of each test epoch.'''
        self.test_loss.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        x, y = batch

        y_hat = self.model(x)

        loss = self.criterion(y_hat.squeeze(), y.squeeze())


        self.test_loss(loss)

        # update and log metrics
        self.log("test_loss", self.test_loss.compute(), on_step=False, on_epoch=True)
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):

        x, y = batch

        y_hat = self.model(x)
        return y_hat, y

    def setup(self, stage: str) -> None:
        """
        This hook is called on every process when using DDP.
        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.opt_params is not None:
            optimizer = self.optimizer(params=self.parameters(), lr = self.lr, **self.opt_params)
            print('Using optimizer with params:', self.opt_params)
        else:
            optimizer = self.optimizer(params=self.parameters(), lr = self.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
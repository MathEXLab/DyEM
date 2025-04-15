import torch
import lightning as L
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import Accuracy


class ClsLitModel(L.LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            **kwargs
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()


    def forward(self, x):
        return self.net(x)


    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()


    def model_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = (logits > 0.5).float()
        return loss, preds, y


    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train_loss", self.train_loss.compute(), on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc.compute(), on_step=False, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val_loss", self.val_loss.compute(), on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc.compute(), on_step=False, on_epoch=True)
        return loss
    

    # def on_validation_batch_end(self):
    #     acc = self.val_acc.compute()
    #     self.val_acc_best(acc)
    #     self.log("val_acc_best", self.val_acc_best.compute())
    

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test_loss", self.test_loss.compute(), on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc.compute(), on_step=False, on_epoch=True)
        return loss, preds, targets
    

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self.forward(x)
        preds = (logits > 0.5).float()
        return preds, y


    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer


if __name__ == "__main__":
    _ = ClsLitModel(None, None, None, None)
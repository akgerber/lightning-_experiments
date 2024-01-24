import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType
import torch
from torch.utils.data import random_split, DataLoader
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy


class MyLightningCLI(LightningCLI):
    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        print(args)
        super().parse_arguments(parser, args)
        print(self.config)

    def after_fit(self):
        print("lol")
        print(self.config)


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
# setup data
# dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
# train_loader = utils.data.DataLoader(dataset)
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
# trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
# trainer.fit(model=autoencoder, train_dataloaders=train_loader)
# load checkpoint
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(
#     checkpoint, encoder=encoder, decoder=decoder
# )

# choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()

# embed 4 fake images!
# fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)


class MNISTLitModule(L.LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """method used to define our model parameters"""
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        # loss
        self.loss = CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        """method used for inference input -> output"""

        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # let's do 3 x (linear + relu)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self):
        """defines model optimizer"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, "multiclass", num_classes=10)
        return preds, loss, acc


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.mnist_test = MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )
        self.mnist_predict = MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )
        mnist_full = MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


def cli_main():
    # define any number of nn.Modules (or use your current ones)
    # encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    # decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    # autoencoder = LitAutoEncoder(encoder, decoder)
    # dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    # dataloader = L.LightningDataModule.from_datasets()

    cli = MyLightningCLI(
        MNISTLitModule, MNISTDataModule, save_config_kwargs={"overwrite": True}
    )
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()

import os
import dotenv
import sys
import warnings

import torch as th
import torch.nn as nn
import lightning as L

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

from absl import app
from absl import flags
from absl import logging


warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=1e-3) # type: ignore
        return optimizer


def main(argv):
    del argv  # Unused.

    dotenv.load_dotenv()
    logging.info("ClearML Test - Training")
    logging.info("Running under Python {0[0]}.{0[1]}.{0[2]}".format(sys.version_info))
    
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    autoencoder = LitAutoEncoder(encoder, decoder)

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = DataLoader(dataset)

    trainer = L.Trainer(limit_train_batches=100, max_epochs=10)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


if __name__ == "__main__":
    app.run(main)

import pytorch_lightning as pl
import torch.nn as nn

class ConvVAE(pl.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #encoded  torch.Size([7, 16, 16, 16])
            nn.Flatten(),
            nn.Linear(16384, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.latent_dim * 2) # batch, 2*latent_dim
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16384),
            nn.Unflatten(dim=1, unflattened_size=(16*2, 16*2, 16)),
            nn.ConvTranspose2d(in_channels=16*2, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Linear(64, 128)
        )

        # Loss functions
        self.mse_loss_fn = nn.MSELoss(reduction='sum')
        self.kl_div_loss_fn = nn.KLDivLoss(reduction='sum')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded

    def training_step(self, batch, batch_idx):
        x = batch
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)

        # Compute reconstruction loss
        mse_loss = self.mse_loss_fn(decoded, x)
        kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = mse_loss + kl_div_loss
        self.log('train_loss', loss)
        self.log('mse_loss', mse_loss)
        self.log('kl_div_loss', kl_div_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

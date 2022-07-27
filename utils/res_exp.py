from .dataset import get_datasets
from models.egnn.resflow import ResCoorFlow
from .utils import create_model, argmax_criterion
from survae.utils import sum_except_batch

import wandb
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import GradScaler, autocast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResCoorExp:
    def __init__(self, config) -> None:
        self.config = config
        self.config['flow'] = "ResCoorFlow" 
        self.config['model'] = ResCoorFlow

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 128

        if "base" not in self.config:
            self.config['base'] = "standard"

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128
        self.train_loader, self.test_loader = get_datasets(type="mqm9", batch_size=self.batch_size)
        self.network, self.optimiser, self.scheduler = create_model(self.config)
        self.base = torch.distributions.Normal(loc=0., scale=1.)
        self.total_logged = 0

    def train(self):
        self.network(torch.zeros(1, 29, 3, device=device), mask=torch.ones(1, 29, device=device, dtype=torch.bool))
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")

        # scaler = GradScaler()
        with wandb.init(project="molecule-flow-3d", config=self.config, entity="iclac") as run:
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = torch.zeros(1, device=device)
                loss_ep_train = torch.zeros(1, device=device)
                self.network.train()

                for idx, batch_data in enumerate(self.train_loader):
                    
                    input = batch_data.pos.to(device)
                    mask = batch_data.mask.to(device)

                    self.optimiser.zero_grad()
                    
                    with autocast(enabled=False):
                        z, log_det = self.network(input, mask=mask)
                        
                    log_prob = sum_except_batch(self.base.log_prob(z) * mask.unsqueeze(2))
                    loss = argmax_criterion(log_prob, log_det)

                    if (loss > 1e3 and epoch > 5) or torch.isnan(loss):
                        if self.total_logged < 30:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'input': input,
                            }, f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            wandb.save(f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            self.total_logged += 1


                    loss_step += loss.detach()
                    loss_ep_train += loss.detach()


                    loss.backward()

                    nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                    self.optimiser.step()

                    step += 1
                    if idx % 10 == 0:
                        ll = (loss_step / 10.).item()
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)

                        loss_step = 0

                if self.scheduler is not None:
                    self.scheduler.step()
                    wandb.log({"Learning Rate/Epoch": self.scheduler.get_last_lr()[0]})

                wandb.log({"NLL/Epoch": (loss_ep_train / len(self.train_loader)).item()}, step=epoch)
                if self.config['upload']:
                    if epoch % self.config['upload_interval'] == 0:
                        if self.scheduler is not None:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimiser.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            }, f"model_checkpoint_{run.id}_{epoch}.pt")
                        else:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimiser.state_dict(),
                            }, f"model_checkpoint_{run.id}_{epoch}.pt")
                        
                        wandb.save(f"model_checkpoint_{run.id}_{epoch}.pt")
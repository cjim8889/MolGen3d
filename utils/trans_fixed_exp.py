from distutils.command.config import config
from .dataset import get_datasets
from models.pos import TransformerCoorFlow
from .utils import create_model, argmax_criterion
from survae.utils import sum_except_batch

import wandb
import torch
import numpy as np
from torch import nn
from einops import rearrange
from torch.cuda.amp import GradScaler, autocast
import math
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def remove_mean_with_constraint(x, size_constraint):
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x

@torch.jit.script
def center_gravity_zero_gaussian_log_likelihood_with_constraint(x, size_constraint):
    B, N_embedded, D = x.size()
    r2 = torch.sum(x.pow(2), dim=[1, 2])

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (size_constraint - 1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * math.log(2 * math.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


class TransCoorFixedExp:
    def __init__(self, config) -> None:
        self.config = config
        self.config['flow'] = "TransformerCoorFlowFixed" 
        self.config['model'] = TransformerCoorFlow

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 32

        if "base" not in self.config:
            self.config['base'] = "standard"

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128
        self.train_loader, self.test_loader = get_datasets(type="mqm9", batch_size=self.batch_size, size_constraint=self.config['size_constraint'])
        
        self.network, self.optimiser, self.scheduler = create_model(self.config)
        self.base = torch.distributions.Normal(loc=0., scale=1.)
        self.total_logged = 0

    def train(self):
        batch_data = next(iter(self.train_loader))

        pos = rearrange(batch_data.pos, "b n d -> b d n").to(device)
        # mask = batch_data.mask.unsqueeze(1).to(device)

        self.network(pos)
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")

        scaler = GradScaler()
        with wandb.init(project="molecule-flow-3d", config=self.config, entity="iclac") as run:
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = 0
                loss_ep_train = 0
                loss_step_count = 0
                self.network.train()

                for idx, batch_data in enumerate(self.train_loader):
                    
                    input = rearrange(batch_data.pos, "b n d -> b d n").to(device)

                    self.optimiser.zero_grad(set_to_none=True)
                    
                    with autocast(enabled=self.config['autocast']):
                        z, log_det = self.network(input)
                        log_prob = None


                    if self.config['base'] == "invariant":
                        z = rearrange(z, "b d n -> b n d")
                        zero_mean_z = remove_mean_with_constraint(z, self.config['size_constraint'])
                        log_prob = sum_except_batch(center_gravity_zero_gaussian_log_likelihood_with_constraint(zero_mean_z, self.config['size_constraint']))
                    else:
                        log_prob = sum_except_batch(self.base.log_prob(z))

                    loss = argmax_criterion(log_prob, log_det)

                    if (loss > 1e3 and epoch > 5) or torch.isnan(loss):
                        if self.total_logged < 30:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'input': input,
                            # 'mask': mask,
                            }, f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            wandb.save(f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            self.total_logged += 1


                    loss_step += loss.detach()
                    loss_ep_train += loss.detach()
                    loss_step_count += 1

                    if self.config['autocast']:
                        scaler.scale(loss).backward()
                        scaler.unscale_(self.optimiser)
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
                        scaler.step(self.optimiser)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2)
                        self.optimiser.step()
        

                    step += 1
                    if loss_step_count % 10 == 0:
                        ll = (loss_step / 10.).item()
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)
                        print(ll)
                        loss_step = 0
                        loss_step_count = 0

                if self.scheduler is not None:
                    self.scheduler.step()
                    wandb.log({"Learning Rate/Epoch": self.scheduler.get_last_lr()[0]})

                # gc.collect()
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
from .dataset import get_datasets
from models.pos.flow import ConditionalTransformerCoorFlow
from .utils import create_model, argmax_criterion
from survae.utils import sum_except_batch
from larsflow.distributions import ResampledGaussian
from models.pos.distro_base import BaseNet
from models.classifier import PosClassifier

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
def remove_mean_with_constraint(x: torch.Tensor, size_constraint: int=18):
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x

@torch.jit.script
def center_gravity_zero_gaussian_log_likelihood_with_constraint(x: torch.Tensor, size_constraint: int=18):
    B, N_embedded, D = x.size()
    r2 = torch.sum(x.pow(2), dim=[1, 2])
    degrees_of_freedom = (size_constraint - 1) * D

    log_normalizing_constant = -0.5 * degrees_of_freedom * math.log(2 * math.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


class ConditionalTransCoorFixedExp:
    def __init__(self, config) -> None:
        self.config = config
        self.config['flow'] = "ConditionalTransformerCoorFlow" 

        self.config['model'] = ConditionalTransformerCoorFlow

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 32

        if "base" not in self.config:
            self.config['base'] = "standard"
            self.base = torch.distributions.Normal(loc=0., scale=1.)

        if self.config['base'] == "resampled":
            net = BaseNet(
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers_transformer'],
                max_nodes=self.config['size_constraint'],
                n_dim=3,
            ).to(device)

            self.base = ResampledGaussian(
                d=self.config['size_constraint'] * 3,
                a=net,
                T=100,
                eps=0.1,
                trainable=True
            ).to(device)
        elif self.config['base'] == "invariant":
            self.base = torch.distributions.Normal(loc=0., scale=1.)
        else:
            self.base = torch.distributions.Normal(loc=0., scale=1.)

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128
        self.train_loader, self.test_loader = get_datasets(type="mqm9", batch_size=self.batch_size, size_constraint=self.config['size_constraint'])
        
        self.network, self.optimiser, self.scheduler = create_model(self.config)

        if self.config['warm_up']:
            self.warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimiser,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.config['warm_up_iters'],
            )

        self.total_logged = 0

        if self.config['two_stage']:
            self.classifier = PosClassifier(feats_dim=64, hidden_dim=256, gnn_size=5)
            self.classifier.load_state_dict(torch.load(config['classifier'], map_location=device)['model_state_dict'])

            self.classifier = self.classifier.to(device)

            for param in self.classifier.parameters():
                param.requires_grad = False


    def train(self):
        batch_data = next(iter(self.train_loader))

        pos = rearrange(batch_data.pos, "b n d -> b d n").to(device)
        context = batch_data.x.squeeze(2).long().to(device)

        self.network(pos, context)
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
                    context = batch_data.x.squeeze(2).long().to(device)

                    self.optimiser.zero_grad(set_to_none=True)
                    
                    with autocast(enabled=self.config['autocast']):
                        z, log_det = self.network(input, context)
                        log_prob = None


                    if self.config['base'] == "invariant":
                        z = rearrange(z, "b d n -> b n d")
                        zero_mean_z = remove_mean_with_constraint(z, self.config['size_constraint'])
                        log_prob = sum_except_batch(center_gravity_zero_gaussian_log_likelihood_with_constraint(zero_mean_z, self.config['size_constraint']))
                    elif self.config['base'] == "resampled":
                        log_prob = sum_except_batch(self.base.log_prob(rearrange(z, "b d n -> b (d n)")))
                    else:
                        log_prob = sum_except_batch(self.base.log_prob(z))

                    loss = argmax_criterion(log_prob, log_det)

                    if idx % self.config['two_stage_step'] == 0 and epoch >= self.config['warmup_epochs']:
                        if self.config['two_stage']:
                            if self.config['two_stage_mode'] == "maxnll":
                                with torch.no_grad():
                                    if self.config['base'] == "resampled":
                                        z, _ = self.base.forward(num_samples=input.shape[0])
                                        z = rearrange(z, "b (d n) -> b d n", d=3)
                                    elif self.config['base'] == "invariant":
                                        z = torch.randn(input.shape[0], self.config['size_constraint'], 3, device=device)
                                        # z = self.base.sample(sample_shape=(input.shape[0], self.config['size_constraint'], 3))
                                        z = remove_mean_with_constraint(z, self.config['size_constraint'])
                                        z = rearrange(z, "b d n -> b n d")

                                    pos, _ = self.network.inverse(z)
                                    pos = rearrange(pos, "b d n -> b n d")
                                    
                                    pos = torch.cat([pos, torch.zeros(pos.shape[0], 29 - self.config['size_constraint'], pos.shape[2], device=device)], dim=1)
                                    
                                    mask = torch.ones(pos.shape[0], 29, device=device, dtype=torch.bool)
                                    mask[:, self.config['size_constraint']:] = False

                                    output = torch.sigmoid(self.classifier(pos, mask=mask)).squeeze()


                                    pos_invalid = pos[output < 0.5]
                                    pos_invalid = rearrange(pos_invalid[:, :self.config['size_constraint'], :], "b n d -> b d n")

                                    wandb.log({"Invalid": pos_invalid.shape[0] * 1.0 / input.shape[0]}, step=step)

                                z, log_det = self.network(pos_invalid)

                                if self.config['base'] == "resampled":
                                    log_prob = sum_except_batch(self.base.log_prob(rearrange(z, "b d n -> b (d n)")))
                                elif self.config['base'] == "invariant":
                                    z = remove_mean_with_constraint(z, self.config['size_constraint'])
                                    log_prob = sum_except_batch(center_gravity_zero_gaussian_log_likelihood_with_constraint(zero_mean_z, self.config['size_constraint']))
                                
                                max_nll= -argmax_criterion(log_prob, log_det)
                            elif self.config['two_stage_mode'] == "prob":
                                if self.config['base'] == "resampled":
                                    z, _ = self.base.forward(num_samples=input.shape[0])
                                    z = rearrange(z, "b (d n) -> b d n", d=3)
                                elif self.config['base'] == "invariant":
                                    z = torch.randn(input.shape[0], self.config['size_constraint'], 3, device=device)
                                    # z = self.base.sample(sample_shape=(input.shape[0], self.config['size_constraint'], 3))
                                    z = remove_mean_with_constraint(z, self.config['size_constraint'])
                                    z = rearrange(z, "b d n -> b n d")

                                pos, _ = self.network.inverse(z)
                                pos = rearrange(pos, "b d n -> b n d")
                                
                                pos = torch.cat([pos, torch.zeros(pos.shape[0], 29 - self.config['size_constraint'], pos.shape[2], device=device)], dim=1)
                                
                                mask = torch.ones(pos.shape[0], 29, device=device, dtype=torch.bool)
                                mask[:, self.config['size_constraint']:] = False

                                output = torch.sigmoid(self.classifier(pos, mask=mask)).squeeze().sum()

                                max_nll = -output

                                
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

                    if idx % self.config['two_stage_step'] == 0 and epoch >= self.config['warmup_epochs']:
                        if self.config['two_stage']:
                            if self.config['two_stage_mode'] == "maxnll":
                                wandb.log({"epoch": epoch, "MaxNLL": max_nll.item()}, step=step)
                            elif self.config['two_stage_mode'] == "prob":
                                wandb.log({"epoch": epoch, "MeanProb": max_nll.item()}, step=step)

                            loss += max_nll
                        

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

                    if self.config['warm_up']:
                        self.warm_up_scheduler.step()

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
                            'base': self.base.state_dict() if self.config['base'] == "resampled" else 0.,
                            }, f"model_checkpoint_{run.id}_{epoch}.pt")
                        else:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimiser.state_dict(),
                            'base': self.base.state_dict() if self.config['base'] == "resampled" else 0.,
                            }, f"model_checkpoint_{run.id}_{epoch}.pt")
                        
                        wandb.save(f"model_checkpoint_{run.id}_{epoch}.pt")
from .dataset import get_datasets
from models import CoorFlow
from models.classifier import PosClassifier
from .utils import create_model, argmax_criterion
from survae.utils import sum_except_batch

import wandb
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import GradScaler, autocast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @torch.jit.script
def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    # assert len(x.size()) == 3
    node_mask = node_mask.unsqueeze(2)
    B, N_embedded, D = x.size()
    # assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


class TwoStageCoorExp:
    def __init__(self, config) -> None:
        self.config = config
        self.config['flow'] = "TwoStageCoorFlow" 
        self.config['model'] = CoorFlow

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 128

        if "base" not in self.config:
            self.config['base'] = "standard"

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128
        self.train_loader, self.test_loader = get_datasets(type="mqm9", batch_size=self.batch_size)
        self.network, self.optimiser, self.scheduler = create_model(self.config)

        self.classifier = PosClassifier(feats_dim=64, hidden_dim=256, gnn_size=5)
        self.classifier.load_state_dict(torch.load(config['classifier'], map_location=device)['model_state_dict'])

        self.classifier = self.classifier.to(device)

        for param in self.classifier.parameters():
            param.requires_grad = False

        # self.base = torch.distributions.Normal(loc=torch.tensor(0, device=device), scale=torch.tensor(1, device=device))
        self.total_logged = 0

    def train(self):
        self.network(torch.zeros(1, 29, 3, device=device), mask=torch.ones(1, 29, device=device, dtype=torch.bool))
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")

        scaler = GradScaler()
        with wandb.init(project="molecule-flow-3d", config=self.config, entity="iclac") as run:
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = 0
                loss_ep_train = 0

                loss_cl_step = 0
                log_p_step = 0

                self.network.train()
                for idx, batch_data in enumerate(self.train_loader):
                    
                    input = batch_data.pos.to(device)
                    mask = batch_data.mask.to(device)

                    self.optimiser.zero_grad(set_to_none=True)
                    
                    with autocast(enabled=self.config['autocast']):
                        z, log_det = self.network(input, mask=mask)

                        log_prob = None


                    if self.config['base'] == "invariant":
                        z = z * mask.unsqueeze(2)
                        zero_mean_z = remove_mean_with_mask(z, node_mask=mask)
                        log_prob = sum_except_batch(center_gravity_zero_gaussian_log_likelihood_with_mask(zero_mean_z, node_mask=mask))
                    else:
                        log_prob = sum_except_batch(self.base.log_prob(z))

                    
                    # # sample = self.base.sample(sample_shape=(self.batch_size, 29, 3))
                    sample = torch.randn(input.shape[0], 29, 3, device=device)
                    sample = sample * mask.unsqueeze(2)
                    sample = remove_mean_with_mask(sample, node_mask=mask)

                    with autocast(enabled=self.config['autocast']):
                        sample_pos, _ = self.network.inverse(sample, mask=mask)
                        pred = self.classifier(sample_pos, mask=mask)

                    classifier_loss = -torch.sigmoid(pred).sum()
                    log_p = argmax_criterion(log_prob, log_det)

                    if epoch > 40:
                        loss = classifier_loss + log_p
                    else:
                        loss = classifier_loss * 10 + log_p

                    if (loss > 1e3 and epoch > 5) or torch.isnan(loss):
                        if self.total_logged < 30:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'input': input,
                            'mask': mask,
                            }, f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            wandb.save(f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            self.total_logged += 1

                    log_p_step += log_p.detach()
                    loss_step += loss.detach()
                    loss_ep_train += loss.detach()
                    loss_cl_step += classifier_loss.detach()

                    if self.config['autocast']:
                        scaler.scale(loss).backward()
                        scaler.unscale_(self.optimiser)
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
                        scaler.step(self.optimiser)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
                        self.optimiser.step()
        

                    step += 1
                    if idx % 10 == 0:
                        ll = (loss_step / 10.).item()
                        lp = (log_p_step / 10.).item()

                        cl = (loss_cl_step / 10.).item()

                        wandb.log({"epoch": epoch, "Loss": ll, "Log_p": lp, "Classifier_Loss": cl}, step=step)

                        loss_step = 0
                        loss_cl_step = 0
                        log_p_step = 0

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
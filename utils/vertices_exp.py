from .dataset import get_datasets
from models.argmax.atom import AtomFlow
from .utils import create_model, argmax_criterion
from survae.utils import sum_except_batch

import wandb
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from .visualise import plot_data3d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VertExp:
    def __init__(self, config) -> None:
        self.config = config
        self.config['flow'] = "AtomFlow" 
        self.config['model'] = AtomFlow

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 128

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128
        self.train_loader, self.test_loader = get_datasets(type="mqm9", batch_size=self.batch_size)
        self.network, self.optimiser, self.scheduler = create_model(self.config)
        self.base = torch.distributions.Normal(loc=0., scale=1.)
        self.plot_size = 16

    def train(self):
        mask = torch.ones(1, 29, device=device, dtype=torch.bool)
        mask[:, -1] = False

        self.network(
            torch.randint(0, 5, (1, 29), device=device),
            torch.randn(1, 29, 3, device=device),
            mask = mask
        )
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")

        scaler = GradScaler()
        with wandb.init(project="molecule-flow-3d", config=self.config, entity="iclac") as run:
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = 0
                loss_ep_train = 0
                self.network.train()

                for idx, batch_data in enumerate(self.train_loader):
                    
                    x = batch_data.x.to(torch.long).squeeze(2).to(device)
                    pos = batch_data.pos.to(device)
                    mask = batch_data.mask.to(device)

                    self.optimiser.zero_grad(set_to_none=True)

                    with autocast(enabled=self.config['autocast']):
                        z, log_det = self.network(x, pos, mask=mask)

                    log_prob = sum_except_batch(self.base.log_prob(z) * mask.unsqueeze(2))
                    loss = argmax_criterion(log_prob, log_det)

                    if self.config['autocast']:
                        scaler.scale(loss).backward()
                        scaler.unscale_(self.optimiser)
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)

                        scaler.step(self.optimiser)
                        scaler.update()
                    else:
                        loss.backward()

                        nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                        self.optimiser.step()

                    loss_step += loss.detach()
                    loss_ep_train += loss.detach()

                    step += 1
                    if idx % 10 == 0:
                        ll = (loss_step / 5.).item()
                        print(ll)
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)

                        
                        loss_step = 0
                    
                    if step % 400 == 0:
                        with torch.no_grad():
                            z = torch.randn(self.plot_size, 29, 5, device=device)
                            atoms_types, _ = self.network.inverse(z, pos[:self.plot_size], mask=mask[:self.plot_size])
                            
                            atoms_types = atoms_types.to("cpu")
                            pos = batch_data.pos[:self.plot_size]
                            mask = batch_data.mask[:self.plot_size]

                            for idx in range(atoms_types.shape[0]):
                                aty = atoms_types[idx].view(-1).numpy()
                                p = pos[idx].view(-1, 3).numpy()
                                
                                plot_data3d(
                                    positions=p,
                                    atom_type=aty,
                                    spheres_3d=False,
                                    save_path=f"{run.id}_{epoch}_{step}_{idx}.png"
                                )

                            wandb.log(
                                {
                                    "epoch": epoch,
                                    "image": [
                                        wandb.Image(f"{run.id}_{epoch}_{step}_{idx}.png") for idx in range(atoms_types.shape[0])
                                    ]
                                },
                                step=step
                            )


                self.scheduler.step()
                wandb.log({"Learning Rate/Epoch": self.scheduler.get_last_lr()[0]})
                wandb.log({"NLL/Epoch": (loss_ep_train / len(self.train_loader)).item()}, step=epoch)
                if self.config['upload']:
                    if epoch % self.config['upload_interval'] == 0:
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        }, f"model_checkpoint_{run.id}_{epoch}.pt")
                    
                        wandb.save(f"model_checkpoint_{run.id}_{epoch}.pt")


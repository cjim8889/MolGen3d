from .dataset import get_datasets
from models import CoorFlow
from .utils import create_model, argmax_criterion


import wandb
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoorExp:
    def __init__(self, config) -> None:
        self.config = config
        self.config['flow'] = "CoorFlow" 
        self.config['model'] = CoorFlow

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 128

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128
        self.train_loader, self.test_loader = get_datasets(type="mqm9", batch_size=self.batch_size)
        self.network, self.optimiser, self.scheduler = create_model(self.config)
        self.base = torch.distributions.Normal(loc=0., scale=1.)

    def train(self):
        self.network(torch.zeros(1, 9, 3, device=device).long())
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")

        with wandb.init(project="molecule-flow-3d", config=self.config):
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = 0
                loss_ep_train = 0
                self.network.train()

                for idx, batch_data in enumerate(self.train_loader):
                    
                    input = batch_data.pos.to(device)

                    self.optimiser.zero_grad()

                    z, log_det = self.network(input)
                    log_prob = torch.sum(self.base.log_prob(z), dim=[1, 2])

                    loss = argmax_criterion(log_prob, log_det)
                    loss.backward()

                    self.optimiser.step()

                    loss_step += loss
                    loss_ep_train += loss

                    
                    step += 1
                    if idx % 5 == 0:
                        ll = (loss_step / 5.).item()
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)

                        
                        loss_step = 0

                self.scheduler.step()
                wandb.log({"NLL/Epoch": (loss_ep_train / len(self.train_loader)).item()}, step=epoch)
                if self.config['upload']:
                    if epoch % self.config['upload_interval'] == 0:
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        }, f"model_checkpoint_{epoch}.pt")
                    
                        wandb.save(f"model_checkpoint_{epoch}.pt")


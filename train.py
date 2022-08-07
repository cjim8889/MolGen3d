import argparse
from utils import CoorExp, VertExp, TwoStageCoorExp, ResCoorExp, TransCoorExp, TransCoorFixedExp, NSFCoorFixedExp
import torch
from torch import batch_norm, nn

parser = argparse.ArgumentParser(description="Molecular Generation MSc Project: 3D")
parser.add_argument("--type", help="Type of experiments e.g. argmaxadj")
parser.add_argument("--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("--batch_size", help="Batch size", type=int, default=128)

parser.add_argument("--hidden_dim", help="Hidden dimension", type=int, default=64)
parser.add_argument("--block_size", help="Block length t parameter", type=int, default=12)
parser.add_argument("--gnn_size", help="Gnn size", type=int, default=2)
parser.add_argument("--base", help="Base distribution", type=str, default="invariant")
parser.add_argument("--num_layers", help="Number of layers in transformer", type=int, default=4)
parser.add_argument("--permute", help="Stochastic permute", type=int, default=1)
parser.add_argument("--scale", help="Scale", type=int, default=1)

parser.add_argument("--conv1x1", help="Conv1x1", type=int, default=1)
parser.add_argument("--conv1x1_node_wise", help="Conv1x1 node wise", type=int, default=0)
parser.add_argument("--partition_size", help="Partition size", type=int, default=9)
parser.add_argument("--size_constraint", help="Size constraint", type=int, default=18)
parser.add_argument("--batch_norm", help="Batch norm", type=int, default=1)
parser.add_argument("--act_norm", help="Act norm", type=int, default=1)

parser.add_argument("--optimiser", help="Optimiser", type=str, default="Adam")
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-03)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-06)
parser.add_argument("--warmup_epochs", help="Warmup epochs", type=int, default=50)

parser.add_argument("--scheduler", help="Scheduler", type=str, default="StepLR")
parser.add_argument("--scheduler_step", help="Scheduler step", type=int, default=3)
parser.add_argument("--scheduler_gamma", help="Scheduler gamma", type=float, default=0.96)
parser.add_argument("--two_stage_step", help="Two stage step", type=int, default=3)

parser.add_argument("--squeeze", help="Apply Squeeze Flow", type=int, default=0)

parser.add_argument("--upload", help="Upload to wandb", type=bool, default=False)
parser.add_argument("--upload_interval", help="Upload to wandb every n epochs", type=int, default=10)

parser.add_argument("--autocast", help="Autocast", type=int, default=0)
parser.add_argument("--loadfrom", help="Load from checkpoint", type=str, default=None)
parser.add_argument("--no_opt", help="No optimiser", type=int, default=0)

parser.add_argument("--num_bins", help="Number of bins", type=int, default=128)

parser.add_argument("--two_stage", help="Two stage", type=int, default=1)

parser.add_argument("--encoder_size", help="Encoder Size for Vert Net", type=int, default=2)
parser.add_argument("--classifier", help="Classifier", type=str, default=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, nn.LazyLinear):
        return

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":
    args = parser.parse_args()

    exp = None

    if args.type  == "coor":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            gnn_size=args.gnn_size,
            base=args.base,
            loadfrom=args.loadfrom,
            autocast=args.autocast != 0,
            no_opt=args.no_opt == 0
        )

        exp = CoorExp(config=config)
    if args.type  == "vert":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            gnn_size=args.gnn_size,
            autocast=args.autocast != 0,
            loadfrom=args.loadfrom,
            no_opt=args.no_opt == 0,
            encoder_size=args.encoder_size,
            permute=args.permute == 1
        )

        exp = VertExp(config=config)
    
    if args.type == "2stage":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            gnn_size=args.gnn_size,
            base=args.base,
            loadfrom=args.loadfrom,
            autocast=args.autocast != 0,
            no_opt=args.no_opt == 0,
            classifier=args.classifier
        )

        exp = TwoStageCoorExp(config=config)
    if args.type  == "res":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            base=args.base,
            loadfrom=args.loadfrom,
            autocast=args.autocast != 0,
            no_opt=args.no_opt == 0
        )

        exp = ResCoorExp(config=config)

    if args.type  == "trans":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            base=args.base,
            loadfrom=args.loadfrom,
            num_layers_transformer=args.num_layers,
            autocast=args.autocast != 0,
            no_opt=args.no_opt == 0
        )

        exp = TransCoorExp(config=config)
    if args.type  == "trans_fixed":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            base=args.base,
            loadfrom=args.loadfrom,
            num_layers_transformer=args.num_layers,
            autocast=args.autocast != 0,
            no_opt=args.no_opt == 0,
            conv1x1=args.conv1x1 == 1,
            conv1x1_node_wise=args.conv1x1_node_wise == 1,
            partition_size=args.partition_size,
            size_constraint=args.size_constraint,
            batch_norm=args.batch_norm == 1,
            act_norm=args.act_norm == 1,
            scale=args.scale == 1,
            two_stage=args.two_stage == 1,
            classifier=args.classifier,
            warmup_epochs=args.warmup_epochs,
            two_stage_step=args.two_stage_step,
            squeeze=args.squeeze == 1,
        )

        exp = TransCoorFixedExp(config=config)
    if args.type  == "spline":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            base=args.base,
            loadfrom=args.loadfrom,
            autocast=args.autocast != 0,
            no_opt=args.no_opt == 0,
            size_constraint=args.size_constraint,
            batch_norm=args.batch_norm == 1,
            num_bins=args.num_bins,
        )

        exp = NSFCoorFixedExp(config=config)
    exp.train()
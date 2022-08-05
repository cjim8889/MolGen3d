import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def argmax_criterion(log_prob, log_det):
    return - torch.mean(log_prob + log_det)

def create_model(config):
    if config['flow'] == "CoorFlow" or config['flow'] == "TwoStageCoorFlow":
        model = config['model'](hidden_dim=config['hidden_dim'], gnn_size=config['gnn_size'], block_size=config['block_size'])
        model = model.to(device)
    elif config['flow'] == "SplineCoorFlow":
        model = config['model'](
            hidden_dim=config['hidden_dim'], 
            block_size=config['block_size'], 
            max_nodes=config['size_constraint'],
        )
    elif config['flow'] == "TransformerCoorFlow":
        model = config['model'](
            hidden_dim=config['hidden_dim'], 
            block_size=config['block_size'], 
            max_nodes=29,
            conv1x1=False,
            batch_norm=False,
            partition_size=(2, 1),
            num_layers_transformer=config['num_layers_transformer'],
        )
        model = model.to(device)
    elif config['flow'] == "TransformerCoorFlowFixed":
        model = config['model'](
            hidden_dim=config['hidden_dim'], 
            block_size=config['block_size'], 
            max_nodes=config['size_constraint'],
            conv1x1=config['conv1x1'],
            partition_size=(1, config['partition_size']),
            num_layers_transformer=config['num_layers_transformer'],
            conv1x1_node_wise=config['conv1x1_node_wise'],
            batch_norm=config['batch_norm'],
        )
        model = model.to(device)
    elif config['flow'] == "AtomFlow":
        model = config['model'](
            hidden_dim=config['hidden_dim'], 
            block_size=config['block_size'], 
            encoder_size=config['encoder_size'],
            gnn_size=config['gnn_size'],
            num_classes=5,
            stochastic_permute=config['permute'],
        )
        model = model.to(device)
    elif config['flow'] == "ResCoorFlow":
        model = config['model'](hidden_dim=config['hidden_dim'], block_size=config['block_size'])
        model = model.to(device)
    if "weight_init" in config:
        model.apply(config["weight_init"])


    if config['optimiser'] == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], amsgrad=True)
    elif config["optimiser"] == "AdamW":
        optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimiser"] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Unknown optimiser: {config['optimiser']}")

    scheduler = None
    if config["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=config['scheduler_step'], gamma=config["scheduler_gamma"])

    
    if config['loadfrom'] is not None:
        states = torch.load(config["loadfrom"], map_location=device)

        model.load_state_dict(states["model_state_dict"])

        if config['no_opt'] is False:
            optimiser.load_state_dict(states["optimizer_state_dict"])
            if "scheduler" in config:
                scheduler.load_state_dict(states["scheduler_state_dict"])

    return model, optimiser, scheduler

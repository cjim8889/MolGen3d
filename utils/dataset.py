from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from .loader import ModifiedDenseDataLoader
import torch
from .qm9 import ModifiedQM9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ToDenseAdjV2(BaseTransform):
    def __init__(self, num_nodes=None):
        self.num_nodes = num_nodes

    def __call__(self, data):
        assert data.edge_index is not None

        orig_num_nodes = data.num_nodes

        if self.num_nodes is None:
            num_nodes = orig_num_nodes
        else:
            assert orig_num_nodes <= self.num_nodes
            num_nodes = self.num_nodes

        if data.edge_attr is None:
            edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        else:
            edge_attr = data.edge_attr

        size = torch.Size([num_nodes, num_nodes] + list(edge_attr.size())[1:])
        # adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)

        # data.orig_adj = adj.to_dense().float()
        
        # tmp = torch.ones(data.orig_adj.shape[0], data.orig_adj.shape[1], 1) * 0.5

        # data.orig_adj = torch.cat((data.orig_adj, tmp), dim=-1)

        # data.adj = data.orig_adj.argmax(dim=-1)
        # data.b_adj = data.adj.clone()
        # data.b_adj[data.b_adj > 0.] = 1.

        # data.adj = data.adj[torch.triu_indices(9,9).unbind()].long()


        data.edge_index = None
        data.edge_attr = None
        data.z = None # Added to use the QM9 dataset

        data.mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.mask[:orig_num_nodes] = 1

        
        if data.x is not None:
            size = [num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

            data.x = data.x[..., :6]

            categorical = data.x[..., :-1]
            categorical = categorical.argmax(dim=-1, keepdim=True).long()

            data.x = categorical

        if data.pos is not None:
            size = [num_nodes - data.pos.size(0)] + list(data.pos.size())[1:]
            data.pos = torch.cat([data.pos, data.pos.new_zeros(size)], dim=0)

        if data.y is not None and (data.y.size(0) == orig_num_nodes):
            size = [num_nodes - data.y.size(0)] + list(data.y.size())[1:]
            data.y = torch.cat([data.y, data.y.new_zeros(size)], dim=0)

        return data

    def __repr__(self) -> str:
        if self.num_nodes is None:
            return super().__repr__()
        return f'{self.__class__.__name__}(num_nodes={self.num_nodes})'

class FilterSize:
    def __init__(self, size_constraint):
        self.size_constraint = size_constraint

    def __call__(self, data):
        return data.N == self.size_constraint

def get_datasets(type="mqm9", batch_size=128, shuffle=True, num_workers=4, size_constraint=None):

    if type == "mqm9" and size_constraint is None:
        # Modified QM9 Dataset where all hydrogen atoms are removed
        # Max_num_nodes: 9
        transform = T.Compose([ToDenseAdjV2(num_nodes=29)])
        dataset = ModifiedQM9(root="./mqm9-datasets", pre_transform=transform)
    else:
        transform = T.Compose([ToDenseAdjV2(num_nodes=size_constraint)])
        dataset = ModifiedQM9(root=f"./mqm9-datasets-{size_constraint}", pre_transform=transform, pre_filter=FilterSize(size_constraint))


    train_loader = ModifiedDenseDataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_loader = ModifiedDenseDataLoader(dataset[int(len(dataset) * 0.8):], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_datasets(type="mqm9")
    
    data = next(iter(train_loader))

    print(data.adj[0])
    print(torch.all(data.adj[0] == data.adj[0].t()))
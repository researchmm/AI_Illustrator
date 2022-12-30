import torch


class LatentMapping_C2S(torch.nn.Module):
    def __init__(self, rrdb_num=5, mlp_layer_num_per_rrdb=5):
        super(LatentMapping_C2S, self).__init__()
        self.rrdb_blocks = torch.nn.ModuleList()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.Linear(512, 512)
        )
        for i in range(rrdb_num):
            self.rrdb_blocks.append(
                torch.nn.Sequential(
                    RRDB_MLP(mlp_layer_num_per_rrdb),
                    torch.nn.Dropout(0.1)
                )
            )
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.Linear(512, 512)
        )

    def forward(self, x):
        x = self.head(x)
        for block in self.rrdb_blocks:
            x = block(x) + x
        x = self.tail(x)
        return x


class LatentMapping_C2S_FCbaseline(torch.nn.Module):
    def __init__(self, fc_number=54):
        super(LatentMapping_C2S_FCbaseline, self).__init__()
        self.body = torch.nn.ModuleList()
        for i in range(fc_number):
            self.body.append(torch.nn.Linear(512, 512))
            self.body.append(torch.nn.PReLU())

    def forward(self, x):
        for layer in self.body:
            x = layer(x)
        return x


class RRDB_MLP(torch.nn.Module):
    def __init__(self, mlp_layer_num=5):
        super(RRDB_MLP, self).__init__()
        self.mlp_layers = torch.nn.ModuleList()
        for i in range(mlp_layer_num):
            self.mlp_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(512 + i * 512, 512),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.PReLU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.PReLU()
                )
            )

    def forward(self, x):
        skip_features = [x]
        for block in self.mlp_layers:
            in_feature = None
            for skip_feature in skip_features:
                if in_feature is None:
                    in_feature = skip_feature
                else:
                    in_feature = torch.cat([in_feature, skip_feature], dim=1)
            out_feature = block(in_feature)
            skip_features.append(out_feature)
        return skip_features[-1]

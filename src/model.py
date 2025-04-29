import torch
import torch.nn as nn

class TradeHorizonScanModel(nn.Module):
    def __init__(
        self,
        n_hs: int,
        #n_yr: int,
        dim_trd: int,
        dim_exp: int,
        dim_imp: int,
        dim_cty: int,
        dropout_p: float = 0.2,
        emb_hs: int = 16
        #emb_yr: int = 16
    ) -> None:
        super().__init__()
        self.hs_emb = nn.Embedding(n_hs, emb_hs)
        #self.yr_emb = nn.Embedding(n_yr, emb_yr)

        self.trd_net = nn.Sequential(
            nn.Linear(dim_trd, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.exp_net = nn.Sequential(
            nn.Linear(dim_exp, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.imp_net = nn.Sequential(
            nn.Linear(dim_imp, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.cty_net = nn.Sequential(
            nn.Linear(dim_cty, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

        total_dim = 32 + 8 + 8 + 4 + emb_hs #+ emb_yr
        self.concat_model_head = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        hs_idx: torch.Tensor,
        #yr_idx: torch.Tensor,
        trd_x: torch.Tensor,
        exp_x: torch.Tensor,
        imp_x: torch.Tensor,
        cty_x: torch.Tensor
    ) -> torch.Tensor:
        hs_emb = self.hs_emb(hs_idx)
        #yr_emb = self.yr_emb(yr_idx)
        trd_out = self.trd_net(trd_x)
        exp_out = self.exp_net(exp_x)
        imp_out = self.imp_net(imp_x)
        cty_out = self.cty_net(cty_x)
        x     = torch.cat([trd_out, exp_out, imp_out, cty_out, hs_emb], dim=1)#yr_emb removed
        return self.concat_model_head(x).squeeze(1)















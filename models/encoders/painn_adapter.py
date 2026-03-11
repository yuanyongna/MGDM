import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from ..common import GaussianSmearing


class Dense(nn.Module):
    """Lightweight linear + optional activation."""

    def __init__(self, in_dim, out_dim, activation=None, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation

    def forward(self, x):
        x = self.lin(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CosineCutoff(nn.Module):
    """Cosine cutoff used in SchNet/PaiNN style models."""

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        x = distances / self.cutoff
        x = torch.clamp(x, 0.0, 1.0)
        return 0.5 * (torch.cos(torch.pi * x) + 1.0) * (x < 1.0)


class PaiNNInteraction(nn.Module):
    """Interaction block: mixes scalar q and vector mu along edges."""

    def __init__(self, n_atom_basis, activation):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.interatomic_context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(self, q, mu, Wij, dir_ij, idx_i, idx_j, n_atoms):
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim=0, dim_size=n_atoms)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim=0, dim_size=n_atoms)

        q = q + dq
        mu = mu + dmu
        return q, mu


class PaiNNMixing(nn.Module):
    """Intra-atomic mixing block."""

    def __init__(self, n_atom_basis, activation, epsilon=1e-8):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.intraatomic_context_net = nn.Sequential(
            Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = Dense(n_atom_basis, 2 * n_atom_basis, activation=None, bias=False)
        self.epsilon = epsilon

    def forward(self, q, mu):
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V ** 2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W
        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class PaiNNEncoder(nn.Module):
    """PaiNN adapter matching EGNN_Sparse_Network interface (scheme B)."""

    def __init__(
        self,
        n_layers,
        feats_input_dim,
        feats_dim,
        edge_attr_dim,
        cutoff,
        num_rbf=64,
        shared_filters=True,
        activation=F.silu,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.feats_dim = feats_dim
        self.shared_filters = shared_filters
        self.edge_attr_dim = edge_attr_dim
        self.cutoff_fn = CosineCutoff(cutoff)
        self.radial_basis = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=num_rbf)
        edge_proj_dim = num_rbf if edge_attr_dim > 0 else 0
        filter_in_dim = num_rbf + edge_proj_dim
        self.filter_in_dim = filter_in_dim
        self.edge_proj = Dense(edge_attr_dim, edge_proj_dim, activation=None) if edge_attr_dim > 0 else None
        # if shared_filters=False, produce layer-specific filters (n_layers * 3 * feats_dim)
        if shared_filters:
            self.filter_net = Dense(filter_in_dim, 3 * feats_dim, activation=None)
        else:
            self.filter_net = Dense(filter_in_dim, n_layers * 3 * feats_dim, activation=None)
        self.interactions = nn.ModuleList([
            PaiNNInteraction(n_atom_basis=feats_dim, activation=activation)
            for _ in range(n_layers)
        ])
        self.mixings = nn.ModuleList([
            PaiNNMixing(n_atom_basis=feats_dim, activation=activation)
            for _ in range(n_layers)
        ])
        self.in_mlp = Dense(feats_input_dim, feats_dim, activation=activation)
        self.mu_proj = nn.Linear(feats_dim, 1, bias=False)
        # small init to keep coordinate updates stable at start
        nn.init.constant_(self.mu_proj.weight, 0.0)
        # stabilize node outputs
        self.out_norm = nn.LayerNorm(feats_dim)

    def forward(self, z, pos, edge_index, edge_attr=None, batch=None, ligand_batch=None, context=None, linker_mask=None):
        idx_i, idx_j = edge_index
        r_ij = pos[idx_j] - pos[idx_i]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True).clamp(min=1e-8)
        dir_ij = r_ij / d_ij

        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)
        filter_input = phi_ij
        if self.edge_proj is not None and edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
            edge_feat = torch.nan_to_num(edge_feat, nan=0.0, posinf=0.0, neginf=0.0)
            filter_input = torch.cat([phi_ij, edge_feat], dim=-1)
        filters = self.filter_net(filter_input) * fcut
        filters = torch.nan_to_num(filters, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-5.0, max=5.0)
        if self.shared_filters:
            filter_list = [filters] * self.n_layers  # list of tensors (E, 3*F)
        else:
            filters = filters.view(-1, self.n_layers, 3 * self.feats_dim)  # (E, L, 3F)
            filter_list = filters.unbind(dim=1)  # tuple length L, each (E, 3F)

        # If upstream features already at feats_dim, skip projection; otherwise project to feats_dim
        if z.size(-1) != self.feats_dim:
            q = self.in_mlp(z).unsqueeze(1)
        else:
            q = z.unsqueeze(1)
        mu = torch.zeros((z.size(0), 3, self.feats_dim), device=z.device, dtype=z.dtype)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixings)):
            Wij = filter_list[i]
            if Wij.dim() == 2:
                Wij = Wij.unsqueeze(1)  # (E,1,3F) to match xj
            q, mu = interaction(q, mu, Wij, dir_ij, idx_i, idx_j, z.size(0))
            q, mu = mixing(q, mu)

        node_attr = q.squeeze(1)
        node_attr = torch.nan_to_num(node_attr, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-5.0, max=5.0)
        node_attr = self.out_norm(node_attr)
        node_attr = torch.tanh(node_attr)
        node_attr = torch.nan_to_num(node_attr, nan=0.0, posinf=0.0, neginf=0.0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        mu_node = scatter_add(mu[idx_j], idx_i, dim=0, dim_size=z.size(0))
        mu_node = torch.nan_to_num(mu_node, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-5.0, max=5.0)
        pos_attr = torch.matmul(mu_node, self.mu_proj.weight.T).squeeze(-1)
        pos_attr = torch.nan_to_num(pos_attr, nan=0.0, posinf=0.0, neginf=0.0)
        pos_attr = pos_attr.clamp(min=-5.0, max=5.0)
        if linker_mask is not None:
            pos_attr = pos_attr * linker_mask.unsqueeze(-1)

        # debug guardrails: zero-out NaNs/Infs and log once
        if torch.isnan(node_attr).any() or torch.isinf(node_attr).any():
            print("[PaiNNAdapter] NaN/Inf in node_attr; zeroing out")
            node_attr = torch.nan_to_num(node_attr, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(pos_attr).any() or torch.isinf(pos_attr).any():
            print("[PaiNNAdapter] NaN/Inf in pos_attr; zeroing out")
            pos_attr = torch.nan_to_num(pos_attr, nan=0.0, posinf=0.0, neginf=0.0)
        return node_attr, pos_attr

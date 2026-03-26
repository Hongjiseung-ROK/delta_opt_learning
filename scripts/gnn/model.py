"""
Edge-Conditioned Message Passing Neural Network (EC-MPNN)

결합 길이 보정량(dft_length - mmff_length)을 엣지 레벨에서 예측한다.

아키텍처:
  1. 노드 임베딩: elem + hybridization → 학습 가능한 임베딩 벡터
  2. 엣지 임베딩: edge_attr → Linear projection
  3. Message Passing (3 레이어):
       메시지: m_ij = MLP_msg(h_j || e_ij)
       집계: a_i = sum(m_ij for j in N(i))
       업데이트: h_i = GRU(a_i, h_i)
  4. 엣지 예측:
       pred_ij = MLP_out(h_i || h_j || e_ij)  → scalar delta

입력 차원:
  - 노드: elem_idx (int), hyb_idx (int) → 임베딩 dim=16 each → concat 32
  - 엣지: [bo, in_ring, ring_size_norm, mmff_len, ang1_norm, ang2_norm] → dim=6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from scripts.gnn.dataset import NUM_ELEM, NUM_HYB


class EdgeConditionedConv(MessagePassing):
    """
    한 번의 메시지 패싱 레이어.
    메시지: m_ij = MLP(h_j || e_ij)
    업데이트: h_i_new = GRU(aggregate(m_*i), h_i)
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="sum")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: [E, node_dim], edge_attr: [E, edge_dim]
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        # aggr_out: [N, hidden_dim], x: [N, node_dim]
        return self.gru(aggr_out, x)


class BondLengthGNN(nn.Module):
    """
    MMFF → DFT 결합 길이 보정량 예측 GNN

    Args:
        node_emb_dim: 원소/혼성화 각각의 임베딩 차원
        edge_hidden:  엣지 projection hidden 차원
        hidden_dim:   메시지 패싱 hidden 차원
        n_layers:     메시지 패싱 반복 횟수
        dropout:      dropout 비율
    """

    def __init__(
        self,
        node_emb_dim: int = 16,
        edge_hidden: int = 32,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        # 노드 임베딩
        self.elem_emb = nn.Embedding(NUM_ELEM + 1, node_emb_dim, padding_idx=NUM_ELEM)
        self.hyb_emb  = nn.Embedding(NUM_HYB  + 1, node_emb_dim, padding_idx=NUM_HYB)
        node_dim = node_emb_dim * 2  # 32

        # 엣지 feature projection: [6] → [edge_hidden]
        raw_edge_dim = 6
        self.edge_proj = nn.Sequential(
            nn.Linear(raw_edge_dim, edge_hidden),
            nn.ReLU(),
        )

        # 메시지 패싱 레이어들
        self.convs = nn.ModuleList([
            EdgeConditionedConv(node_dim, edge_hidden, hidden_dim)
            for _ in range(n_layers)
        ])

        # 엣지 레벨 예측 헤드: (h_i || h_j || e_ij) → delta
        pred_in_dim = node_dim * 2 + edge_hidden  # 32+32+32=96
        self.pred_head = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:          [N, 2]  long tensor (elem_idx, hyb_idx)
            edge_index: [2, E]  long tensor
            edge_attr:  [E, 6]  float tensor

        Returns:
            pred: [E]  예측된 delta_length (각 엣지)
        """
        # 1. 노드 임베딩
        h = torch.cat([
            self.elem_emb(x[:, 0]),
            self.hyb_emb(x[:, 1]),
        ], dim=-1)  # [N, 32]

        # 2. 엣지 임베딩
        e = self.edge_proj(edge_attr)  # [E, 32]

        # 3. 메시지 패싱
        for conv in self.convs:
            h_new = conv(h, edge_index, e)
            h = h + h_new  # residual

        # 4. 엣지 레벨 예측
        i_idx = edge_index[0]  # source atom
        j_idx = edge_index[1]  # target atom
        h_i = h[i_idx]  # [E, 32]
        h_j = h[j_idx]  # [E, 32]

        pred = self.pred_head(torch.cat([h_i, h_j, e], dim=-1))  # [E, 1]
        return pred.squeeze(-1)  # [E]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BondLengthGNN()
    print(f"모델 파라미터 수: {count_parameters(model):,}")
    print(model)

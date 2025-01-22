import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

from timm.models.layers import DropPath

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class RecGLARE(SequentialRecommender):
    def __init__(self, config, dataset):
        super(RecGLARE, self).__init__(config, dataset)
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.drop_path = config["drop_path"]

        self.seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.num_heads = config["num_heads"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.GLARE_block = nn.ModuleList([
            GLARE(
                d_model=self.hidden_size,
                seq_len=self.seq_len,
                num_heads=self.num_heads,
                drop=self.dropout_prob,
                drop_path=self.drop_path,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.GLARE_block[i](item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )
        return scores

class GLARE(nn.Module):
    def __init__(self, d_model, seq_len, num_heads, drop, drop_path, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.attn = LinearAttentionWithRope(
            d_model=d_model,
            seq_len=seq_len,
            num_heads=num_heads,
        )
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop)
        self.LayerNorm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(d_model, eps=1e-12)

        self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model * 4), act_layer=nn.GELU, drop=drop)
        self.act_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.num_layers == 1:
            x = self.LayerNorm1(x)
            act_res = self.act(self.act_proj(x))

            x = self.in_proj(x)
            x = self.attn(x)

            x = self.drop_path(self.out_proj(x * act_res))

            x = x + self.mlp(self.LayerNorm2(x))
        else:
            shortcut = x
            x = self.LayerNorm1(x)

            act_res = self.act(self.act_proj(x))

            x = self.in_proj(x)
            x = self.attn(x)

            x = self.out_proj(x * act_res)
            x = shortcut + self.drop_path(x)

            x = x + self.mlp(self.LayerNorm2(x))
        return x


class LinearAttentionWithRope(nn.Module):
    def __init__(self, d_model, seq_len, num_heads):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads

        self.qk = nn.Linear(d_model, d_model * 2, bias=True)
        self.elu = nn.ELU()

        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model, bias=True)
        self.rope = RoPE(seq_len, d_model)

    def forward(self, x):
        B, L, D = x.shape
        num_heads = self.num_heads
        head_dim = D // num_heads

        qk = self.qk(x).reshape(B, L, 2, D).permute(2, 0, 1, 3)  # [2, B, L, D]
        q, k, v = qk[0], qk[1], x  # [B, L, D]

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q_rope = self.rope(q).reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
        k_rope = self.rope(k).reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
        k = k.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # [B, num_heads, L, 1]
        kv = (k_rope.transpose(-2, -1) * (L ** -0.5)) @ (v * (L ** -0.5))  # [B, num_heads, L, head_dim]
        x = q_rope @ kv * z  # [B, num_heads, L, head_dim]

        x = x.transpose(1, 2).reshape(B, L, D)
        v = v.transpose(1, 2).reshape(B, L, D).permute(0, 2, 1)  # [B, D, L] for conv1d

        # Local Shortcut
        if causal_conv1d_fn is None:
            v_output = self.conv1d(v)[..., :L]
        else:
            # temporal conv1d with CUDA optimization
            v_output = causal_conv1d_fn(
                x=v,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
            ).permute(0, 2, 1)  # [B, L, D]
        x = x + v_output

        return x

class RoPE(torch.nn.Module):
    def __init__(self, seq_len, d_model, base=10000):
        super().__init__()

        k_max = d_model // 2
        theta = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.outer(torch.arange(seq_len), theta)

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
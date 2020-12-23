import torch
import torch.nn as nn
import math


class Config:
    """Hyperparameters from GPT1 paper"""
    # better not mess with these
    vocab_size  = 40000 # 40_000 BPE merges
    p_drop      = 0.1 # residual, embedding, and attention dropout probability
    activation  = nn.GELU() # activation function in feedforward sublayer
    n_decoder   = 12 # sequential decoder layers
    weight_mean = 0.0
    weight_std  = 0.02
    def __init__(self, **kwargs):
        self.emb_dim     = 768 # embedding size
        self.n_head      = 12 # attention heads
        assert self.emb_dim % self.n_head == 0
        self.head_dim    = self.emb_dim // self.n_head
        self.ff_dim      = self.emb_dim * 4
        self.max_len     = 512 # sequence length
        self.batch_size  = 4 # more didn't fit in GPU
        for keyword, value in kwargs.items():
            setattr(self, keyword, value)


class Attention(nn.Module):
    """Masked multihead self-attention, which is also scaled btw"""
    def __init__(self, config):
        super().__init__()
        self.config         = config
        self.n_head         = config.n_head
        self.head_dim       = config.head_dim
        self.key            = nn.Linear(config.emb_dim, config.emb_dim)
        self.query          = nn.Linear(config.emb_dim, config.emb_dim)
        self.value          = nn.Linear(config.emb_dim, config.emb_dim)
        self.attn_dropout   = nn.Dropout(config.p_drop)
        self.output_dropout = nn.Dropout(config.p_drop)
        self.project        = nn.Linear(config.emb_dim, config.emb_dim)
        mask                = torch.tril(torch.ones(config.max_len, config.max_len))
        # when chopped to multiple heads tensor becomes 4-dimensional
        self.register_buffer('mask', mask.view(1, 1, config.max_len, config.max_len))
    def forward(self, x):
        b, t, d = x.size()
        # (batch, time, emb_dim) -> (batch, n_head, time, head_dim)
        k    =   self.key(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q    = self.query(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v    = self.value(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1)
        # scale
        attn = attn * ( 1.0 / math.sqrt(k.size(-1)) )
        # crop mask to shorter sequence length if needed
        attn = attn.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y    = attn @ v
        # concat heads back to original (batch, time, emb_dim)
        y    = y.transpose(1, 2).contiguous().view(b, t, d)
        y    = self.project(y)
        # apply dropout to the output of each sub-layer
        # before it is added to the sub-layer input and normalized
        y    = self.output_dropout(y)
        return y


class FeedForward(nn.Module):
    """Feed Forward layer"""
    def __init__(self, config):
        super().__init__()
        self.activation      = config.activation
        self.make_it_bigger  = nn.Linear(config.emb_dim, config.ff_dim)
        self.make_it_smaller = nn.Linear(config.ff_dim, config.emb_dim)
        self.output_dropout  = nn.Dropout(config.p_drop)
    def forward(self, x):
        x = self.make_it_bigger(x)
        x = self.activation(x)
        x = self.make_it_smaller(x)
        x = self.output_dropout(x)
        return x


class Decoder(nn.Module):
    """Decoder layer"""
    def __init__(self, config):
        super().__init__()
        self.norm1         = nn.LayerNorm(config.emb_dim)
        self.norm2         = nn.LayerNorm(config.emb_dim)
        self.attn          = Attention(config)
        self.feedforward   = FeedForward(config)
    def forward(self, x):
        # LayerNorm place was allegedly changed in GPT2 paper
        attn_output        = self.attn( self.norm1(x) )
        x                  = x + attn_output
        feedforward_output = self.feedforward( self.norm2(x) )
        x                  = x + feedforward_output
        return x


class GPT(nn.Module):
    """Hopefuly GPT-like model"""
    def __init__(self, config):
        super().__init__()
        # 512 tokens in sequence
        self.max_len     = config.max_len
        self.mean        = config.weight_mean
        self.std         = config.weight_std
        # 768 features
        self.embed       = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, config.max_len, config.emb_dim))
        self.emb_dropout = nn.Dropout(config.p_drop)
        # 12 sequential decoder layers
        self.decoder     = nn.Sequential(*[Decoder(config) for _ in range(config.n_decoder)])
        self.norm        = nn.LayerNorm(config.emb_dim)
        self.head        = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.apply(self.init_weights)
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=self.mean, std=self.std)
            # allegedly best practice for bias initialization
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, y=None):
        # infer batch size and sequence length
        b, t   = x.size()
        assert t <= self.max_len, f"Sequence too long, max length is{self.max_len}"
        # regular embedings and positional embeddings cropped to sequence length
        x      = self.embed(x) + self.pos_embed[:, :t, :]
        x      = self.emb_dropout(x)
        x      = self.decoder(x)
        x      = self.norm(x)
        logits = self.head(x)
        return logits


class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, data, config):
        self.data      = data
        self.max_len   = config.max_len
    def __len__(self):
        return len(self.data) - self.max_len
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.max_len + 1]
        x = torch.tensor(seq[:-1]).long()
        y = torch.tensor(seq[1:]).long()
        return x, y

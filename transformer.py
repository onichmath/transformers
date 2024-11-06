# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
from torch.nn import functional as F 

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

"""
1. Encoder
Encoder should output sequence of embeddings for each word in input sequence
- provide mean of embeddings across sequence dimension to linear classifier
2. Feedforward classifier
- receives the output from transformer encoder and makes predictions
- one hidden layer 
3. Training 
- Loss updates weights of both encoder and classifier through backpropagation simultaneously
"""
# Encoder self attention attends to all tokens
# Decoder self attention attends autoregressively to all tokens before the current token
class SelfAttentionBase(nn.Module):
    def __init__(self, embed_dim, block_size, head_dim, autoregression, dropout):
        super().__init__()
        self.embed_dim = embed_dim 
        self.head_dim = head_dim
        self.block_size = block_size
        self.autoregression = autoregression

        # Channels x Head size
        self.query_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.key_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.value_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False)

        if self.autoregression:
            self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = nn.Dropout(dropout)

    def get_shape(self, x):
        B, T, C = x.shape
        assert T == self.block_size
        assert C == self.embed_dim
        return B, T, C

    def project_linear_components(self, x):
        # Get the linearly projected queries, keys, and values
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        return query, key, value

    def mask_causal_logits(self, logits, T):
        # Mask out future tokens if decoder
        if self.autoregression:
            logits = logits.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        return logits

    def compute_attention(self, logits, value):
        weights = F.softmax(logits, dim=-1)
        regularized_weights = self.dropout(weights)
        attention = torch.einsum("btt,btc->btc", regularized_weights, value)
        return attention, weights


class SelfAttentionHead(SelfAttentionBase):
    # Single head of attention based off "Let's build GPT: from scratch, in code, spelled out" by Andrej Karpathy
    # and the "Attention is All You Need" paper
    def __init__(self, embed_dim, block_size, head_dim, autoregression, dropout):
        super().__init__(embed_dim=embed_dim, 
                         block_size=block_size, 
                         head_dim=head_dim, 
                         autoregression=autoregression, 
                         dropout=dropout)

    def forward(self, x):
        # Forward pass for basic self attention
        B, T, C = self.get_shape(x)
        q, k, v = self.project_linear_components(x)

        # Compute attention weights
        logits = torch.einsum("btc,bTc->bTt", q, k) # B x T x C @ B x T x C -> B x T x T
        logits = logits / (C ** 0.5) # Divide by sqrt(d_k) to prevent peaky softmax

        logits = self.mask_causal_logits(logits, T)
        attention, weights = self.compute_attention(logits, v)

        return attention, weights 

class AlibiAttentionHead(SelfAttentionBase):
    # https://arxiv.org/pdf/2108.12409v2
    def __init__(self, embed_dim, block_size, head_dim, autoregression, slope_bias, dropout):
        super().__init__(embed_dim=embed_dim,
                         block_size=block_size,
                         head_dim=head_dim,
                         autoregression=autoregression,
                         dropout=dropout)

        linear_bias = torch.arange(self.block_size).float().unsqueeze(1) - torch.arange(self.block_size).float()
        linear_bias = linear_bias - 2 * torch.tril(linear_bias).float()
        linear_bias *= slope_bias
        self.register_buffer("linear_bias", linear_bias)

    def forward(self, x):
        B, T, C = self.get_shape(x)

        q, k, v = self.project_linear_components(x)

        # Compute attention weights
        logits = torch.einsum("btc,bTc->bTt", q, k) # B x T x C @ B x T x C -> B x T x T
        logits += self.linear_bias[:T, :T]
        logits = self.mask_causal_logits(logits, T)

        attention, weights = self.compute_attention(logits, v)
        return attention, weights 

class BasicBigBirdAttentionHead(SelfAttentionBase):
    # https://arxiv.org/pdf/2007.14062v2
    def __init__(self, embed_dim, block_size, head_dim, autoregression, dropout):
        super().__init__(embed_dim=embed_dim,
                         block_size=block_size,
                         head_dim=head_dim,
                         autoregression=autoregression,
                         dropout=dropout)

        # Global attention attends to CLS token
        big_bird_mask = self.create_big_bird_mask(self.block_size, 1, self.block_size // 8, self.block_size // 16)
        self.register_buffer("attention_mask", big_bird_mask)

    def create_big_bird_mask(self, block_size, num_global, num_local, num_random):
        mask = torch.zeros(block_size, block_size)
        for i in range(num_global):
            # Global attention
            mask[i, :] = 1
            mask[:, i] = 1
        for i in range(block_size):
            # Local attention
            start = max(0, i - num_local)
            end = min(block_size, i + num_local + 1)
            mask[i, start:end] = 1
            # Random attention
            random_indices = torch.randperm(block_size)[:num_random]
            mask[i, random_indices] = 1
        return mask

    def forward(self, x):
        B, T, C = self.get_shape(x)
        q, k, v = self.project_linear_components(x)

        # Compute attention weights
        logits = torch.einsum("btc,bTc->bTt", q, k) # B x T x C @ B x T x C -> B x T x T
        logits = logits / (self.embed_dim ** 0.5) # Divide by sqrt(d_k) to prevent peaky softmax
        logits = logits.masked_fill(self.attention_mask == 0, float("-inf")) # Mask out attention weights
        logits = self.mask_causal_logits(logits, T)

        attention, weights = self.compute_attention(logits, v)
        return attention, weights 
        
class MultiHeadAttention(nn.Module):
    # Multiple heads of attention based off "Let's build GPT: from scratch, in code, spelled out" by Andrej Karpathy
    # and the "Attention is All You Need" paper
    def __init__(self, num_heads, embed_dim, block_size, autoregression, attention, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self._init_heads(num_heads, embed_dim, block_size, autoregression, attention, dropout)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _init_heads(self, num_heads, embed_dim, block_size, autoregression, attention, dropout):
        head_dim = embed_dim // num_heads
        params = {
            "embed_dim": embed_dim,
            "block_size": block_size,
            "head_dim": head_dim,
            "autoregression": autoregression,
            "dropout": dropout
            }
        if attention == "basic":
            self.heads = nn.ModuleList([SelfAttentionHead(**params) for _ in range(num_heads)])
        elif attention == "alibi":
            slope_biases = torch.tensor([2**(i * -8 / num_heads) for i in range(num_heads)], dtype=torch.float)
            self.heads = nn.ModuleList([AlibiAttentionHead(slope_bias=slope_biases[i], **params) for i in range(num_heads)])
        elif attention == "bigbird":
            self.heads = nn.ModuleList([BasicBigBirdAttentionHead(**params) for _ in range(num_heads)])
        else:
            raise ValueError(f"Unrecognized attention type: {attention}")


    def forward(self, x):
        # Multihead attention based off "Attention is All You Need" paper
        attention_maps = []
        attentions = []
        for head in self.heads:
            attention_output, attention_map = head(x)
            attention_maps.append(attention_map)
            attentions.append(attention_output)

        concatenated_attention = torch.cat(attentions, dim=-1)
        concatenated_attention = self.dropout(self.proj(concatenated_attention))
        return concatenated_attention, attention_maps

class FeedForward(nn.Module):
    # Feed forward network based off "Let's build GPT: from scratch, in code, spelled out" by Andrej Karpathy
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.ff_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff_net(x)

class TransformerBlock(nn.Module):
    # Single transformer block based off "Attention is All You Need" paper
    def __init__(self, embed_dim, num_heads, block_size, hidden_dim, autoregression, attention, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, 
                                            embed_dim=embed_dim, 
                                            block_size=block_size, 
                                            autoregression=autoregression, 
                                            attention=attention, 
                                            dropout=dropout)

        self.feed_forward = FeedForward(embed_dim=embed_dim, 
                                        hidden_dim=hidden_dim, 
                                        dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(embed_dim) # Pre-normalization
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Residual connection around each sub-block
        attentions, attention_maps = self.attention(self.layer_norm1(x))
        with torch.no_grad(): # Discard gradients for the attention maps
            attention_maps = [attn_map.detach() for attn_map in attention_maps]
        x = x + attentions
        x = x + self.feed_forward(self.layer_norm2(x))
        return x, attention_maps

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, block_size, embed_dim):
        super().__init__()
        self.positional_embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, x):
        B, T = x.shape
        return self.positional_embedding(torch.arange(T, device=x.device))

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, block_size, embed_dim):
        # Based off implementation of "Attention Is All You Need" position encodings at https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        super().__init__()
        self.embed_dim = embed_dim
        self.block_size = block_size

        # Compute the positional encodings once in log space
        pe = torch.zeros(block_size, embed_dim)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = x.unsqueeze(-1) + self.pe[:x.size(1),  :].unsqueeze(0)
        return pe

class TransformerBase(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, autoregression, attention="basic", position_encoding="learned", dropout=0.0):
        super().__init__()
        # Positional encoding
        if attention == "alibi" and (position_encoding is not None):
            raise ValueError("Alibi attention does not support positional encoding")

        if position_encoding == "learned":
            self.position_encoding = LearnedPositionalEncoding(block_size, embed_dim)
        elif position_encoding == "sinusoidal":
            self.position_encoding = SinusoidalPositionalEncoding(block_size, embed_dim)
        else:
            self.position_encoding = None

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, block_size, hidden_dim, autoregression, attention, dropout) 
            for _ in range(num_layers)
            ])
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            # nn.init.kaiming_normal_(module.weight)
            # Normal distribution reduces train perplexity to be within range, at cost of test perplexity
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def embed(self, x):
        # Return the embeddings for the input sequence
        B, T = x.shape
        token_embeddings = self.token_embedding_table(x)
        if not self.position_encoding:
            return token_embeddings
        pos_embeddings = self.position_encoding(x)
        return token_embeddings + pos_embeddings

class Encoder(TransformerBase):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, attention="basic", position_encoding="learned", dropout=0.0):
        super().__init__(vocab_size=vocab_size,
                         embed_dim=embed_dim,
                         block_size=block_size,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         hidden_dim=hidden_dim,
                         autoregression=False,
                         attention=attention,
                         position_encoding=position_encoding,
                         dropout=dropout)
        self.classifier = nn.Linear(embed_dim, 3)

    def forward(self, x, y=None):
        x = self.embed(x)
        attention_maps = []
        for block in self.blocks:
            x, maps = block(x)
            attention_maps.extend(maps)

        x = self.layer_norm(x)
        x = x.mean(dim=1) # Mean across the time dimension
        logits = self.classifier(x)

        cross_entropy_loss = None
        if y is not None:
            cross_entropy_loss = F.cross_entropy(logits, y)
        return logits, cross_entropy_loss, attention_maps

class Decoder(TransformerBase):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, attention="basic", position_encoding="learned", dropout=0.0):
        super().__init__(vocab_size=vocab_size,
                         embed_dim=embed_dim,
                         block_size=block_size,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         hidden_dim=hidden_dim,
                         autoregression=True,
                         position_encoding=position_encoding,
                         attention=attention,
                         dropout=dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, y=None):
        x = self.embed(x)
        attention_maps = []
        for block in self.blocks:
            x, maps = block(x)
            attention_maps.extend(maps)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        cross_entropy_loss = None
        if y is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            cross_entropy_loss = F.cross_entropy(logits_flat, y_flat)
        return logits, cross_entropy_loss, attention_maps

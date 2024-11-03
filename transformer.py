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
class SelfAttentionHead(nn.Module):
    # Single head of attention based off "Let's build GPT: from scratch, in code, spelled out" by Andrej Karpathy
    # and the "Attention is All You Need" paper
    def __init__(self, embed_dim, block_size, head_dim, autoregression, dropout):
        super().__init__()
        self.embed_dim = embed_dim 
        self.head_dim = head_dim
        self.block_size = block_size
        self.autoregression = autoregression

        # Channels x Head size
        self.query_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.key_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.value_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False) # What is aggregated from the input sequence

        if self.autoregression:
            # Buffer instead of parameter
            self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        assert T == self.block_size
        assert C == self.embed_dim

        # Linearly project queries, keys, and values
        query = self.query_linear(x) # B x T x C
        key = self.key_linear(x) # B x T x C
        value = self.value_linear(x) # B x T x C

        # Compute attention weights
        # logits = torch.einsum("btc,bcT->btt", query, key.transpose(-1, -2)) # B x T x C @ B x C x T -> B x T x T
        logits = torch.einsum("btc,bTc->bTt", query, key) # B x T x C @ B x T x C -> B x T x T
        logits = logits / (self.embed_dim ** 0.5) # Divide by sqrt(d_k) to prevent peaky softmax
        # If decoder, mask out future tokens
        if self.autoregression:
            logits = logits.masked_fill(self.mask[:T, :T] == 0, float("-inf")) # Mask out future tokens if decoder, B x T x T

        weights = F.softmax(logits, dim=-1) # B x T x T
        regularized_weights = self.dropout(weights)

        attention = torch.einsum("btt,btc->btc", regularized_weights, value) # B x T x T @ B x T x C -> B x T x C
        return attention, weights 
        
class MultiHeadAttention(nn.Module):
    # Multiple heads of attention based off "Let's build GPT: from scratch, in code, spelled out" by Andrej Karpathy
    # and the "Attention is All You Need" paper
    def __init__(self, num_heads, embed_dim, block_size, autoregression, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([SelfAttentionHead(embed_dim, block_size, self.head_dim, autoregression, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

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
    def __init__(self, embed_dim, num_heads, block_size, hidden_dim, autoregression, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, block_size, autoregression, dropout)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout)

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

class TransformerBase(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, autoregression, dropout=0.0):
        super().__init__()
        # Embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding_table = nn.Embedding(block_size, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, block_size, hidden_dim, autoregression, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def embed(self, x):
        # Return the embeddings for the input sequence
        B, T = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.pos_embedding_table(torch.arange(T, device=x.device))
        return token_embeddings + pos_embeddings

class Encoder(TransformerBase):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, dropout=0.0):
        super().__init__(vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, False, dropout)
        self.classifier = nn.Linear(embed_dim, 3)

    def forward(self, x, y=None):
        x = self.embed(x)
        attention_maps = []
        for block in self.blocks:
            x, attention_maps = block(x)
            attention_maps.extend(attention_maps)

        x = self.layer_norm(x)
        x = x.mean(dim=1) # Mean across the time dimension
        logits = self.classifier(x)

        cross_entropy_loss = None
        if y is not None:
            cross_entropy_loss = F.cross_entropy(logits, y)
        return logits, cross_entropy_loss, attention_maps

class Decoder(TransformerBase):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, dropout=0.0):
        super().__init__(vocab_size, embed_dim, block_size, num_heads, num_layers, hidden_dim, True, dropout)
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, y=None):
        x = self.embed(x)
        attention_maps = []
        for block in self.blocks:
            x, attention_maps = block(x)
            attention_maps.extend(attention_maps)

        x = self.layer_norm(x)
        logits = self.classifier(x)

        cross_entropy_loss = None
        if y is not None:
            cross_entropy_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, cross_entropy_loss, attention_maps

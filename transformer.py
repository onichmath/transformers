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
    def __init__(self, embed_dim, block_size, head_dim, is_decoder=False):
        super().__init__()
        self.embed_dim = embed_dim 
        self.head_dim = head_dim
        self.block_size = block_size
        self.is_decoder = is_decoder

        # Channels x Head size
        self.query_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.key_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.value_linear = nn.Linear(self.embed_dim, self.head_dim, bias=False) # What is aggregated from the input sequence

        if self.is_decoder:
            # Buffer instead of parameter
            self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)))


    def forward(self, x):
        B, T, C = x.shape
        assert T == self.block_size
        assert C == self.embed_dim

        # Linearly project queries, keys, and values
        query = self.query_linear(x) # B x T x C
        key = self.key_linear(x) # B x T x C
        value = self.value_linear(x) # B x T x C

        # Compute attention weights
        logits = torch.einsum("btc,bcT->bth", query, key.transpose(-1, -2)) # B x T x C @ B x C x T -> B x T x T
        logits = logits / (self.embed_dim ** 0.5) # Divide by sqrt(d_k) to prevent peaky softmax
        # If decoder, mask out future tokens
        if self.is_decoder:
            logits = logits.masked_fill(self.mask[:T, :T] == 0, float("-inf")) # Mask out future tokens if decoder, B x T x T
        weights = F.softmax(logits, dim=-1) # B x T x T

        attention = torch.einsum("btt,btc->btc", weights, value) # B x T x T @ B x T x C -> B x T x C
        return attention
        
class MultiHeadAttention(nn.Module):
    # Multiple heads of attention based off "Let's build GPT: from scratch, in code, spelled out" by Andrej Karpathy
    # and the "Attention is All You Need" paper
    def __init__(self, num_heads, embed_dim, block_size, is_decoder=False):
        super(MultiHeadAttention).__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(embed_dim, block_size, self.head_dim, is_decoder) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class LanguageModel(nn.Module):
    # Batch x Time x Channel 
    # logs.shape = B x T x C
    # crossentropyLoss needs B x C x T
    def __init__(self, vocab_size):
        super(LanguageModel).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, x):
        # Based loosely off of the transformer architecture from the Attention is All You Need paper
        # Combined with the implementation of "Let's Build GPT: from scratch, in code, spelled out"
        B, T, C = x.shape
        token_embeddings = self.token_embedding_table(x) # B x T x C
        pos_embeddings = self.pos_embedding_table(torch.arange(T, device=x.device)) # T x C
        x = token_embeddings + pos_embeddings



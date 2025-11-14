# Training GPT-2 from Scratch
A complete implementation of GPT-2 (124M parameters) trained from scratch, such that loss is less than 0.099999.

## Overview

This implements a GPT-2 model (Generative Pre-trained Transformer 2) trained entirely from scratch on Shakespeare's text. The model learns to predict the next word in a sequence, enabling it to generate Shakespeare-like text.

- Model size: ~124M parameters (GPT-2 small)
- Training dataset: Shakespeare's complete works (~338,025 tokens)
- Architecture: 12 layers, 12 attention heads, 768 embedding dimensions
- Context window: 1024 tokens

## Dataset

The Shakespeare Dataset : We train on the complete works of William Shakespeare, saved as input.txt. This includes all his plays, sonnets, and poems.
- File size: ~1.1 MB

    class DataLoaderLite:
        def __init__(self, B, T):
            self.B = B  # Batch size
            self.T = T  # Sequence length
            
            with open('input.txt', 'r') as f:
                text = f.read()
            
            enc = tiktoken.get_encoding('gpt2')
            self.tokens = torch.tensor(enc.encode(text))

### Dataset Statistics
    ======================================================================
    DATASET DIAGNOSTICS
    ======================================================================
    Total characters: 1,115,394
    Total tokens: 338,025
    Unique tokens in dataset: 11,706 / 50257 total vocab
    
     First 300 characters of data:
    ----------------------------------------------------------------------
    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    
    All:
    We know't, we know't.
    
    First Citizen:
    Let us
    ----------------------------------------------------------------------
    ======================================================================

Once the text is fed into a tokenizer, the dataset becomes a list of tokens. Total tokens: 338,025
BPE encoding reduces size by ~70%, Instead of 1.1M characters, we now have only 338k tokens. BPE compression means fewer steps required to train.

### Unique tokens used: 11,706 out of GPT-2’s 50,257 vocabulary
- GPT-2 has ~50k possible tokens.
- Shakespeare text only uses ~11.7k of them (~23%).
- Unused tokens: 38,551

This means Shakespeare doesn't contain modern words like “internet”, “quantum”, “algorithm”, “yeet”, etc. Many GPT-2 tokens never appear in this dataset. This is typical for small/single-domain datasets.
These are tokens that never appear in Shakespeare’s works.
Examples:

- Modern slang
- Internet terms
- Technical vocabulary
- Non-English words
- Emoticons, symbols, etc.

The model will never learn embeddings for these unused tokens. Their embeddings stay random because they receive zero gradient updates. If we use this GPT-2 model for other domains, it may behave poorly on those words.
For example:
- A Shakespeare-trained GPT-2 will write Shakespearean English well.
- But it will struggle with modern language since those tokens never appear in training.

## Tokenization

### What is Tokenization?
Tokenization converts raw text into numerical tokens that the model can process. Instead of treating each character separately, we use subword tokenization, which breaks text into meaningful chunks.

- Character-level:  "Hello" → ['H', 'e', 'l', 'l', 'o']  (5 tokens)
- Subword-level:    "Hello" → ['Hello']                  (1 token)

Advantages of Subword Tokenization:
- Efficiency: Fewer tokens = faster training and more context
- Vocabulary balance: Not too small (like characters) or too large (like full words)
- Handles rare words: Can break unknown words into known subwords

We use GPT-2's BPE (Byte Pair Encoding) tokenizer from the tiktoken library:

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("To be, or not to be")
    
### GPT-2 Vocabulary:
- Total vocabulary size: 50,257 tokens
- Includes: common words, subwords, punctuation, special characters
- Shakespeare dataset uses: ~6,000-8,000 unique tokens

Example Tokenization:

    Text:     "ROMEO: Wherefore art thou Romeo?"
    Tokens:   [49, 13649, 46, 6350, 1640, 1650, 7009, 32444, 30]

## Model Architecture

    class GPTConfig:
        block_size: int = 1024    # Maximum sequence length
        vocab_size: int = 50257   # Number of possible tokens
        n_layer: int = 12         # Number of transformer blocks
        n_head: int = 12          # Number of attention heads per layer
        n_embd: int = 768         # Embedding dimension


## Architecture Breakdown (bottom-up)

    Loss: Cross-Entropy with next token
       ↑
    Logits (probability distribution over vocabulary)
       ↑
    Output Projection (768 → 50257)
       ↑
    Final Layer Norm
       ↑
    12 Transformer Blocks (each contains):
       ├─ Residual Connection
       ├─ MLP (3072 → 768 → output)
       ├─ Layer Norm
       ├─ Residual Connection
       ├─ Multi-Head Self-Attention (12 heads)
       └─ Layer Norm
       ↑
    [Batch, Sequence, 768] ← [SUM of both embeddings]
       ↑
    Positional Embedding (block_size → n_embd)
       ↑
    Token Embedding (vocab_size → n_embd)
       ↑
    Input: [Batch, Sequence] ← Token IDs

### Parameter Count:

- Token embeddings: 50,257 × 768 = 38.6M
- Position embeddings: 1,024 × 768 = 0.8M
- 12 transformer blocks: ~7M each = 84M

Total: ~124M parameters

## Positional Embeddings

Transformers process tokens in parallel using self-attention, which is invariant to order: the model has no built-in notion of which token came first.
To model word order (“dog bites man” ≠ “man bites dog”), Transformers inject positional information into the input embeddings.

Transformers have used several approaches over time, but GPT-2 specifically uses learned absolute positional embeddings.

### Absolute Sinusoidal Positional Embeddings (Original Transformer, 2017)

The original Attention Is All You Need paper introduced fixed sinusoidal absolute positional embeddings. These are deterministic vectors added to the token embeddings:

      PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
      PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))


### Learned Absolute Positional Embeddings (GPT-2)

GPT-2 keeps the idea of absolute positions (position 0, position 1, …) but does not use sinusoids. Instead, GPT-2 uses a learned embedding table:

    def forward(self, idx):
        B, T = idx.size()
        
        # Token embeddings: "what" is each token
        tok_emb = self.transformer.wte(idx)  # [B, T, 768]
        
        # Position embeddings: "where" is each token
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.transformer.wpe(pos)   # [T, 768]
        
        # Combine: token meaning + position information
        x = tok_emb + pos_emb  # [B, T, 768]

- Each position (0–1023 in GPT-2 small) has its own learned vector.
- These vectors are updated through gradient descent.
- The model learns whatever positional structure is most helpful for the language-modeling task.

GPT-2 is a decoder-only, autoregressive model. Empirically, for language modeling, learned positional embeddings outperform sinusoidal ones because:
- They adapt to natural language statistics
- They better encode start-of-sequence and end-of-sequence patterns
- They integrate smoothly with GPT-2’s block architecture

## Self-Attention Mechanism


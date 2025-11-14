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
Self-attention lets each token "look at" all previous tokens to understand context. 

For example:

        "The cat sat on the mat because it was tired"
                                           ↑
                                          What does "it" refer to?
                                          
The attention mechanism helps the model understand that "it" refers to "cat".

### Query, Key, Value Vectors
Each token is transformed into three vectors:

    kv = self.c_attn(x)  # Linear projection: [B, T, 768] → [B, T, 2304]
    q, k, v = qkv.split(self.n_embd, dim=2)  # Split into Q, K, V
    
**What are Q, K, V?**
- Query (Q): "What am I looking for?"
- Key (K): "What information do I have?"
- Value (V): "What information will I pass forward?" (new representation of the token)

### Multi-Head Attention
Instead of one attention mechanism, GPT-2 uses 12 parallel attention heads:

    n_head = 12
    head_dim = n_embd // n_head  # 768 / 12 = 64

    # Reshape into multiple heads
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, 12, T, 64]
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)
    
**Why Multiple Heads?**
Each head can focus on different aspects (similar to multiple kernels in CNN)

- Head 1: Syntactic relationships (subject-verb agreement)
- Head 2: Semantic relationships (synonyms, antonyms)
- Head 3: Long-range dependencies (pronouns to their referents)
... and so on

### Attention Computation

    # 1. Compute attention scores (how much to focus on each token)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    # Result: [B, n_head, T, T] — every token attends to every token
    
    # 2. Apply causal mask (can only look at past tokens)
    att = att.masked_fill(mask == 0, float('-inf'))
    # Ensures token at position 5 cannot see tokens 6, 7, 8, ...
    
    # 3. Softmax: Convert scores to probabilities
    att = F.softmax(att, dim=-1)  # Sum to 1.0 along last dimension
    
    # 4. Weighted sum of values
    y = att @ v  # [B, n_head, T, head_dim]
    
### Attention Score Calculation:

    score(i, j) = (Q[i] · K[j]) / √64
    
    For token at position 3:
    scores = [0.05, 0.15, 0.30, 0.50, -∞, -∞, ...]
               ↑     ↑     ↑     ↑     ↑    ↑
              pos 0  pos 1 pos 2 pos 3 pos 4 pos 5
                                (current) (future - masked)

    After softmax: [0.10, 0.20, 0.30, 0.40, 0.0, 0.0, ...]

    
### Embedding Size & Heads

- Embedding dimension (n_embd): 768
- Number of heads (n_head): 12
- Dimension per head: 64 (768 ÷ 12)

## Training Configuration

### Optimization

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,              # Learning rate
        betas=(0.9, 0.95),    # Momentum terms
        weight_decay=0.1      # L2 regularization
    )

### Learning Rate Schedule:

    def get_lr(step):
        # 1. Warmup: Linear increase for first 200 steps
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        
        # 2. Cosine decay: Smooth decrease to min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

**Why Warmup?**

- Prevents large gradients at initialization
- Helps stabilize early training

**Why Cosine Decay?**

- Smooth learning rate reduction
- Better final convergence than step decay

### Training Loop

    for step in range(max_steps):
        # 1. Get batch
        x, y = train_loader.next_batch()
        
        # 2. Forward pass
        logits, loss = model(x, y)
        
        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 5. Update weights
        optimizer.step()

### Loss Function (Cross-Entropy Loss)

    loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))

### Training

    using device: cuda
    Model parameters: 124.44M
    loaded 338025 tokens
    1 epoch = 41 batches
    
    ======================================================================
    TRAINING FROM SCRATCH
    ======================================================================
    Target: loss < 0.1
    Max steps: 5000
    Batch size: 32 x 256 = 8,192 tokens/batch
    Dataset: 338,025 tokens (~41 batches per epoch)
    ======================================================================
    
    step    0 | loss: 10.948163 | avg: 10.948163 | best: 10.948163 | lr: 1.50e-06 | dt: 2546ms
    step   10 | loss: 9.169275 | avg: 9.716531 | best: 9.144935 | lr: 1.65e-05 | dt: 29ms
    step   20 | loss: 8.678150 | avg: 8.922493 | best: 8.678150 | lr: 3.15e-05 | dt: 37ms
    step   30 | loss: 8.388718 | avg: 8.504875 | best: 8.359287 | lr: 4.65e-05 | dt: 33ms
    step   40 | loss: 8.000952 | avg: 8.053351 | best: 7.750473 | lr: 6.15e-05 | dt: 41ms
    step   50 | loss: 7.270163 | avg: 7.547222 | best: 7.193012 | lr: 7.65e-05 | dt: 28ms
    step   60 | loss: 6.699990 | avg: 7.080446 | best: 6.699990 | lr: 9.15e-05 | dt: 30ms
    step   70 | loss: 6.385480 | avg: 6.569657 | best: 6.385480 | lr: 1.07e-04 | dt: 32ms
    step   80 | loss: 5.910585 | avg: 6.179581 | best: 5.910585 | lr: 1.21e-04 | dt: 28ms
    step   90 | loss: 5.654613 | avg: 5.951821 | best: 5.654613 | lr: 1.36e-04 | dt: 28ms
    step  100 | loss: 5.758962 | avg: 5.883077 | best: 5.654613 | lr: 1.51e-04 | dt: 28ms
    ....

     Step 1400 summary:
       Current loss: 0.181554
       Best loss: 0.149027
       Avg last 100: 0.264241
       Learning rate: 2.60e-04
    
    step 1410 | loss: 0.167040 | avg: 0.160273 | best: 0.128597 | lr: 2.60e-04 | dt: 28ms
    step 1420 | loss: 0.247891 | avg: 0.165310 | best: 0.128597 | lr: 2.59e-04 | dt: 28ms
    step 1430 | loss: 0.212630 | avg: 0.191252 | best: 0.128597 | lr: 2.59e-04 | dt: 28ms
    step 1440 | loss: 0.139345 | avg: 0.148913 | best: 0.104838 | lr: 2.58e-04 | dt: 41ms
    step 1450 | loss: 0.111260 | avg: 0.133446 | best: 0.104838 | lr: 2.57e-04 | dt: 27ms
    step 1460 | loss: 0.104511 | avg: 0.131384 | best: 0.104511 | lr: 2.57e-04 | dt: 28ms
    step 1470 | loss: 0.129679 | avg: 0.144884 | best: 0.102478 | lr: 2.56e-04 | dt: 28ms
    
     TARGET ACHIEVED at step 1474!
    step 1474 | loss: 0.094124 | best: 0.094124 | lr: 2.56e-04 | dt: 28ms
    Final loss: 0.094124
    
    ======================================================================
    TRAINING COMPLETED
    ======================================================================
    Final loss: 0.094124
    Best loss: 0.094124
    Total steps: 1475
    ======================================================================
    
    Model saved to gpt2_checkpoint.pt


### Training Loss
<img width="518" height="463" alt="image" src="https://github.com/user-attachments/assets/96855120-98fd-4421-a63f-d8e2df596e87" />


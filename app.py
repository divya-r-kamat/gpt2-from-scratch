import gradio as gr
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import tiktoken

# Model Architecture (same as training)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")

model = GPT(GPTConfig())
checkpoint = torch.load('gpt2_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

enc = tiktoken.get_encoding('gpt2')

print(f"✅ Model loaded! Training loss: {checkpoint['loss']:.6f}, Step: {checkpoint['step']}")

# Generation function
@torch.no_grad()
def generate_text(prompt, max_length, temperature, top_k, num_samples):
    """Generate text completions"""

    if not prompt.strip():
        return "Please enter a prompt!"

    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_samples, 1).to(device)

    # Generate
    generated_texts = []
    for _ in range(max_length):
        if tokens.size(1) >= model.config.block_size:
            break

        logits, _ = model(tokens)
        logits = logits[:, -1, :] / temperature

        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    # Decode
    for i in range(num_samples):
        text = enc.decode(tokens[i].tolist())
        generated_texts.append(text)

    return "\n\n" + "="*80 + "\n\n".join(generated_texts)

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Shakespeare GPT") as demo:
    gr.Markdown("""
    # 🎭 Shakespeare GPT - Custom Trained Language Model

    This is a GPT-2 (124M) model trained from scratch on Shakespeare's works.
    Start with a prompt and watch it generate Shakespearean-style text!

    **Training Stats:** Loss: {:.6f} | Steps: {:,}
    """.format(checkpoint['loss'], checkpoint['step']))

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="ROMEO:",
                lines=3,
                value="First Citizen:"
            )

            with gr.Row():
                generate_btn = gr.Button("🎭 Generate", variant="primary", size="lg")
                clear_btn = gr.ClearButton(value="Clear")

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                max_length = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=150,
                    step=10,
                    label="Max Length (tokens to generate)"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature (higher = more creative)"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=40,
                    step=1,
                    label="Top-K (vocabulary filtering)"
                )
                num_samples = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of Samples"
                )

        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=20,
                max_lines=30
            )

    # Example prompts
    gr.Examples(
        examples=[
            ["First Citizen:", 150, 0.8, 40, 3],
            ["ROMEO:", 200, 0.9, 50, 2],
            ["To be or not to be", 150, 0.7, 40, 3],
            ["All the world's a stage", 180, 0.85, 45, 2],
            ["KING:", 200, 0.8, 40, 3],
        ],
        inputs=[prompt_input, max_length, temperature, top_k, num_samples],
        label="📚 Example Prompts (click to load)"
    )

    # Event handlers
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_length, temperature, top_k, num_samples],
        outputs=output_text
    )

    clear_btn.add([prompt_input, output_text])

    gr.Markdown("""
    ---
    ### 💡 Tips:
    - **Start with character names** like "ROMEO:", "JULIET:", or "First Citizen:" for dialogue
    - **Lower temperature** (0.5-0.7) = more coherent, **higher** (0.9-1.2) = more creative
    - **Top-K filtering** controls vocabulary diversity (40-50 works well)
    - Try generating multiple samples to see different continuations!

    ### 📊 Model Info:
    - Architecture: GPT-2 (124M parameters)
    - Training: From scratch on Shakespeare corpus (338k tokens)
    - Final training loss: {:.4f}
    """.format(checkpoint['loss']))

if __name__ == "__main__":
    demo.launch(share=True)

import torch
import torch.nn as nn
from transformers import AutoModel

class RoutingNetwork(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        return weights

class ExpertAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out

class SentimixtureNet(nn.Module):
    def __init__(self, base_model_name='xlm-roberta-base', hidden_dim=768, num_experts=4):
        super().__init__()

        # Try to use local cache to avoid remote downloads in environments like Streamlit Cloud
        try:
            self.encoder = AutoModel.from_pretrained(base_model_name, local_files_only=True)
        except:
            self.encoder = AutoModel.from_pretrained(base_model_name)

        self.routing = RoutingNetwork(hidden_dim, num_experts)
        self.experts = nn.ModuleList([ExpertAttention(hidden_dim) for _ in range(num_experts)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        
        routing_weights = self.routing(x)  # Shape: (batch_size, seq_len, num_experts)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # Shape: (batch_size, seq_len, hidden_dim, num_experts)
        
        routing_weights = routing_weights.unsqueeze(2)  # (batch_size, seq_len, 1, num_experts)
        mixed_output = (expert_outs * routing_weights).sum(-1)  # Weighted sum over experts
        
        pooled = mixed_output.mean(dim=1)  # Mean over sequence length (pooled representation)
        return self.classifier(pooled)

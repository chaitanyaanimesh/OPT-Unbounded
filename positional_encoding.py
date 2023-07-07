import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=8192):
        super(PositionalEncoding, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.register_buffer('position_encoding_cache', self._generate_position_encoding_cache())

    def _generate_position_encoding_cache(self):
        position_encoding_cache = torch.zeros(self.max_len, self.embedding_dim)
        position_idx = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        frequency = torch.exp((torch.arange(0, self.embedding_dim, 2).float()/ self.embedding_dim) * (-torch.log(torch.tensor(10000))))
        position_encoding_cache[:, 0::2] = torch.sin(position_idx * frequency)
        position_encoding_cache[:, 1::2] = torch.cos(position_idx * frequency)
        return position_encoding_cache

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        attention_mask = attention_mask.long()
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        positions = positions[:, past_key_values_length:]
        
        if positions.size(1)>self.max_len:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.max_len = positions.size(1)
            self.register_buffer('position_encoding_cache', self._generate_position_encoding_cache())
            self.position_encoding_cache = self.position_encoding_cache.to(device)
        
        position_encoding = self.position_encoding_cache[positions]
        return position_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # 논문에 sqrt(d_model) 곱해주는걸로 되어있음
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len  # 최대 입력 길이 설정

        PE = torch.zeros(max_len, d_model)  # (max_len x d_model) 의 0 행렬 생성

        # max_len 만큼 나열하고 unsqueeze를 해서 (max_len x 1) 사이즈로 변환. 단순 인덱스 생성임
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  

        # 직관적 코드는 아래와 같으나 연산상의 효율을 원한다면 exp, log 사용. 수식에 log 를 씌워 계산 후 다시 exp로 환원
        # div_term = torch.pow(10000, (torch.arange(0, d_model, 2).float() / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # d_model 만큼의 길이인 텐서에 홀수, 짝수 나눠서 곱해주기. 그걸 각 포지션 마다.
        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)

        # Add an extra dimension for batch size and register the positional encoding as a buffer
        PE = PE.unsqueeze(0).transpose(0, 1) # 여기서 차원 안맞을 수 있음
        self.register_buffer('PE', PE) # self.register_buffer('pe', pe)는 pe 텐서를 모델의 버퍼로 등록하여 학습되지 않지만 저장 및 로드되는 텐서로 만듭니다.

    def forward(self, x):
        # 입력값에 PE값 더해줌
        x = x + self.pe[:x.size(0), :]  # x의 크기 만큼만 잘라서 쓰도록, 차원 안맞을 수 있음 
        return x
        

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        

    def forward(self, query, key, value, mask, dropout):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
            scores = self.dropout(scores)

        attention = (F.softmax(scores, dim=-1)) @ value # @ 는 torch.matmul과 같은 연산자
        return attention



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h # 헤드의 갯수
        assert d_model % h == 0, "d_model is not divisible by h" # d_model 은 h로 나누어 떨어져야 함
        self.dropout = dropout

        self.d_k = d_model / h
        # d_k 만큼을 h 번 하는 것. d_model = h * self.d_k 이나 직관성을 위해 나눠서 적음
        self.w_q = nn.Linear(d_model, h * self.d_k) # Wq
        self.w_k = nn.Linear(d_model, h * self.d_k) # Wk
        self.w_v = nn.Linear(d_model, h * self.d_k) # Wv

    
        self.w_o = nn.Linear(h * self.d_k, d_model) # Wo, h * d_v = d_model, d_v = d_k
    
    def forward(self, query, key, value, mask):
        query = self.w_q(query) # (Batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(key) # (Batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(value) # (Batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # torch의 view 함수는 numpy의 reshape 과 같은 역할을 하나 메모리에 저장을 하지 않음
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) --> ( batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) 

        x = ScaledDotProductAttention()(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # PyTorch의 텐서는 메모리 상에 연속적으로 저장되지 않을 수 있습니다. 
        # contiguous 메서드는 텐서를 연속된 메모리 블록으로 복사하여 새로운 텐서를 만듭니다. 
        # 이는 이후의 view 메서드 호출이 오류 없이 수행될 수 있도록 합니다.

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

# layer normalization 을 최대한 scratch로 구현
class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class AddAndNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=True)
        self.W2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W2(self.dropout(F.relu(self.W1(x)))))  # 두 번 다 드롭아웃 넣는게 맞는지?


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout
        self.self_attention = MultiHeadAttention(self.d_model, self.h, self.dropout)
        self.feed_forward = FeedForward(self.d_model, self.d_ff, self.dropout)
        self.add_and_norm_mh = AddAndNorm()
        self.add_and_norm_ff = AddAndNorm()

    def forward(self, x, mask):
        x = self.add_and_norm_mh(x, self.self_attention(x, x, x, mask))
        x = self.add_and_norm_ff(x, self.feed_forward(x))
        return x
    

class Encoder(nn.Moduel):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        # self.norm = LayerNormalization() # normalization 들어가야하는지? 논문 그림엔 없음 

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x # self.norm(x)
   

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout
        self.self_attention = MultiHeadAttention(self.d_model, self.h, self.dropout)
        self.cross_attention = MultiHeadAttention(self.d_model, self.h, self.dropout)
        self.feed_forward = FeedForward(self.d_model, self.d_ff, self.dropout)
        self.add_and_norm_mmh = AddAndNorm()
        self.add_and_norm_mh = AddAndNorm()
        self.add_and_norm_ff = AddAndNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.add_and_norm_mmh(x, self.self_attention(x, x, x, tgt_mask))
        x = self.add_and_norm_mh(x, self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.add_and_norm_ff(x, self.feed_forward(x))
        return x
        

class Decoder(nn.Moduel):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        # self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x # self.norm(x)


class LinearLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear_layer = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.linear_layer(x), dim = -1) # for numerical stability, using log softmax


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, linear_layer: LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear_layer = linear_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def linear_softmax(self, x):
        return self.linear_layer(x)
    

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the postional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_layer = EncoderLayer(d_model, h, d_ff, dropout)
        encoder_blocks.append(encoder_layer)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_layer = DecoderLayer(d_model, h, d_ff, dropout)
        decoder_blocks.append(decoder_layer)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    linear_layer = LinearLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, linear_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
  































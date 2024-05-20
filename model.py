import torch
import torch.nn as nn
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
        PE = PE.unsqueeze(0).transpose(0, 1)
        self.register_buffer('PE', PE) # self.register_buffer('pe', pe)는 pe 텐서를 모델의 버퍼로 등록하여 학습되지 않지만 저장 및 로드되는 텐서로 만듭니다.

    def forward(self, x):
        # 입력값에 PE값 더해줌
        x = x + self.pe[:x.size(0), :]  # x의 크기 만큼만 잘라서 쓰도록
        return x
        

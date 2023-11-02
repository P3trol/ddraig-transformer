import torch 
import torch.nnw
import math

## d_model is the dimension of the model == 512
## d_ff is the dimension of the feedforward network model == 2048
## d_k is the dimension of the key == 64
## d_v is the dimension of the value == 64
## seq_len is the length of the sequence 
## batch_size is the size of the batch

## h is the number of heads 
## N is the number of encoder/decoder layers 
## q is the query vector
## k is the key vector
## v is the value vector


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int): ## constructor
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) ## embedding layer with vocab_size and d_model, embedding layer is a lookup table


    def forward(self, x)
        return self.embedding(x) * math.sqrt(self.d_model) ## return the embedding layer times the square root of the d_model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:)
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
        ## create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model) ## create a matrix of zeros with seq_len and d_model
        ## create a vector of shape (seq_len, 1)
        position  = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) ## create a vector of zeros with seq_len and d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) ## create a vector of zeros with seq_len and d_model
        ## apply the sin to even positions in the array
        pe[:, 0::2] = torch.sin(position * div_term) ## apply the sin to even positions in the array
        pe[:, 1::2] = torch.cos(position * div_term) ## apply the cos to odd positions in the array

        pe = pe.unsqueeze(0) ## (1, seq_len, d_model) ## unsqueeze means add a dimension to the tensor
        self.register_buffer('pe', pe) ## register_buffer is a tensor that is not a model parameter, but should be part of the modules state

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) ## making the tensor not learn
        return self.dropout(x)

class layerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None: ## eps is a small number to avoid division by zero 
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) ## alpha is a learnable parameter, multiplied
        self.bias = nn.Parameter(torch.zeros(1)) ## bias is a learnable parameter, added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) 
        std = x.std(dim=-1, keepdim=True) ## mean and std are the mean and std of the last dimension of the tensor
        return self.alpha * (x - mean) / (std + self.eps) + self.bias ## return the alpha times the x minus the mean divided by the std plus the bias   

class feedfowardblock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) ## linear layer with d_model and d_ff, (W1) and bias (b1)
        self.dropout = nn.Dropout(dropout) ## dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model)  ## linear layer with d_ff and d_model, (W2) and bias (b2)



    def forward(self, x):
        #  a tensor (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module): 
    def --init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h  
        assert d_model % h == 0, 'd_model is not divisible by h' ## assert is a boolean expression, if it is false, it will raise an error

        self.d_k = d_model // h ## d_k is the dimension of the key == 64
        self.w_q = nn.Linear(d_model, d_model) ## linear layer with d_model and d_model, (W_q)
        self.w_k = nn.Linear(d_model, d_model) ## linear layer with d_model and d_model, (W_k)
        self.w_v = nn.Linear(d_model, d_model) ## linear layer with d_model and d_model, (W_v)

        self.w_o = nn.Linear(d_model, d_model) ## linear layer with d_model and d_model, (W_o)
        self.dropout = nn.Dropout(dropout) ## dropout layer

    @staticmethod ## static method, no need to pass self
    def attention(query,key,value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
        attn_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) ## (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9) ## if the mask is 0, replace it with -1e9(-1000000000)
        attn_score = attn_score.softmax(dim=-1) ## (batch, h, seq_len, seq_len)
        if dropout is not None:
            attn_score = dropout(attn_score)

        return (attn_score @ value), attn_score 



    def forward(self, q, k, v, mask): ## mask is the mask to avoid the padding tokens(0)
        query = self.w_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) 

        ## split the query, key and value into h heads # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k)--> tranpose(batch_size, h, seq_len, d_k,)
        query = query.view(query.shape[0], query.shape[-1] , self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[-1] , self.h, self.d_k).transpose(1, 2) 
        value = value.view(value.shape[0], value.shape[-1] , self.h, self.d_k).transpose(1, 2)

        x, self.attn_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)



        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return seld.w_o(x)


class residualConnect(nn.module): ## add and norm ##
    def __init__ (self, dropout: float) -> None:
        super().__init__()
        self.dropout == nn.Dropout(dropout)
        self.norm = layerNormalization()

    def forward(self, x, sublayer):
        return x self.dropout(sublayer(self.norm(x)))    
        


class EncoderBock(nn.module):

    def __init__(self, self_attenion_block: MultiHeadAttention, feed_foward_block: feedfowardblock, dropout: flase) -> none:
        super().__init__()
        self.self_attenion_block = self_attenion_block
        self.feed_foward_block = feed_foward_block
        self.residualConnect = nn.modulelist([residualConnect(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residualConnect[0](x, lambda x: self.self_attenion_block(x,x,x,src_mask))
        x = self.residualConnect[1](x,self.feed_foward_block)
        return x

class Encoder(nn.module):
    
    def __init__(self, layers: nn.modulelist) -> None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()

    def forward(self, x, mask):
        for layers in self.layers:
            x = layer(x, mask)
        return self.norm(x)




import torch 
import torch.nn as nn
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


    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) ## return the embedding layer times the square root of the d_model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:
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
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
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


class residualConnect(nn.Module): ## add and norm ##
    def __init__ (self, dropout: float) -> None:
        super().__init__()
        self.dropout == nn.Dropout(dropout)
        self.norm = layerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))    
        


class EncoderBock(nn.Module):

    def __init__(self, self_attenion_block: MultiHeadAttention, feed_foward_block: feedfowardblock, dropout: False) -> None:
        super().__init__()
        self.self_attenion_block = self_attenion_block
        self.feed_foward_block = feed_foward_block
        self.residualConnect = nn.modulelist([residualConnect(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residualConnect[0](x, lambda x: self.self_attenion_block(x,x,x,src_mask))
        x = self.residualConnect[1](x,self.feed_foward_block)
        return x

class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()

    def forward(self, x, mask):
        for layers in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    
    def __init__(sefl, self_attenion_block: MultiHeadAttention, cross_attenion_block: MultiHeadAttention, feed_foward_block: feedfowardblock, dropout: float) -> None:
        super().__init__()
        self.self_attenion_block = self_attenion_block
        self.cross_attenion_block = cross_attenion_block
        self.feed_foward_block = feed_foward_block
        self.residualConnect = nn.modulelist([residualConnect(dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.residualConnect[0](x, lambda x: self.self_attenion_block(x,x,x,trg_mask))
        x = self.residualConnect[1](x, lambda x: self.cross_attenion_block(x,enc_out,enc_out,src_mask))
        x = self.residualConnect[2](x,self.feed_foward_block)
        return x

class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()

    def forward(self, x, enc_out, src_mask, trg_mask):
        for layers in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module): ## linear layer ##
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module): ## transformer model ## 

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, trg_embed: InputEmbedding, src_pos: PositionalEncoding, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.trg_pos = trg_pos
        self.src_pos = src_pos
        self.proj_layer = proj_layer

    def enocde(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, trg, enc_out, src_mask, trg_mask): ## basically the forward pass
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, enc_out, src_mask, trg_mask)

    def projection(self, x):
        return self.proj_layer(x)
        
#edit this function to build your transformer model, maybe the src_vocab_size and trg_vocab_size are different

def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_en: int, trg_sqg_len: int, d_model: int = 512, N: int = 6, H: int =8, dropout: float =0.1, d_ff: int = 2048 ) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    trg_embed = InputEmbedding(d_model, trg_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # create the encoder blocks
    encoder_block = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, H, dropout)
        feed_foward_block = feedfowardblock(d_model, d_ff, dropout)
        encoder_block = EncoderBock(encoder_self_attention_block, feed_foward_block, dropout)
        encoder_block.append(encoder_block)

    # create the decoder blocks
    decoder_block = []
    for _ in range(N):
        deocer_self_attention_block = MultiHeadAttention(d_model, H, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, H, dropout)
        feed_foward_block = feedfowardblock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(deocer_self_attention_block, decoder_cross_attention_block, feed_foward_block, dropout)
        decoder_block.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.modulelist(encoder_block))
    decoder = Decoder(nn.modulelist(decoder_block))

    # create the projection layer
    projection_Layer = ProjectionLayer(d_model, trg_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_Layer)

    # initialize the parameters with xavier 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
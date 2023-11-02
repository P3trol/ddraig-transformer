import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

class EnglishToWelshDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        #create the tensors for the special tokens
        self.sos_token = torch.tensor([tokenizer_trg.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_trg.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_trg.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self): #return the length of the dataset
        return len(self.ds)

    def __getitem__(self, idx: Any) -> Any:
        # Get the source and target sentences
        src_target_pair = self.ds[idx]
        src = src_target_pair['translation']['en']
        trg = src_target_pair['translation']['cy']

        # Encode the source and target sentences
        enc_input = self.tokenizer_src.encode(src).ids
        dec_input = self.tokenizer_trg.encode(trg).ids

        # Add the special tokens
        enc_padding = self.seq_len - len(enc_input) - 2 # 2 for the special tokens 
        dec_padding = self.seq_len - len(dec_input) - 1 # 1 for the special tokens

        if enc_input < 0 or dec_input < 0:
            raise ValueError('The sequence length is too long')

        # Add the special tokens SOS and EOS
        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_padding, dtype=torch.int64)
            ]
        )
        # add the special tokens SOS to the decoder input
        dec_input = torch.cat(
            [   
                self.eos_token,
                torch.tensor(dec_input, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_padding, dtype=torch.int64)
            ]
        )

        # Add the special tokens EOS to the label(what we expect the decoder to output)
        label = torch.cat(
            [
                torch.tensor(dec_input, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_padding, dtype=torch.int64)
            ]
        ) 

        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return{
            'enc_input': enc_input, # (seq_len)
            'dec_input': dec_input, # (seq_len)
            'encoder_mask': (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(dec_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            'label': label, # (seq_len)
            'src_text': src_text,
            'trg_text': trg_text
        }

def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal= 1).type(torch.int) # triu = upper triangular part of a matrix
    return mask == 0

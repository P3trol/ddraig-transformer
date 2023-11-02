import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path # For file paths handling

def get_all_sentences(ds, lang):
    for item in ds: #ds is a dataset
        yield item['translation'][lang] #translation is a dict with keys 'en' and 'cy'


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_path'] = 'tokenizer_{}.json'
    tokenizer_path = config['tokenizer_path'].format(lang)
    if not Path.exists(tokenizer_path):
        # Build a new tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token=['UNK'])) #UNK = unknown token
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2) #PAD = padding, SOS = start of sentence, EOS = end of sentence
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus100', 'cy-en', split='train')

    #build tokenizers
    Tokenizer_src = get_or_build_tokenizer(config, ds_raw, 'en')#en = English
    Tokenizer_src = get_or_build_tokenizer(config, ds_raw, 'cy')#cy = Welsh

    #keep 90% of the data for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])





import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import EnglishToWelshDataset, casual_mask
from Model import build_transformer

from config import get_config, get_weights_file_name


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter # For logging to Tensorboard 

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

    train_ds = EnglishToWelshDataset(train_ds_raw, Tokenizer_src, Tokenizer_trg, config['en'], config['cy'], config['seq_len'])
    val_ds = EnglishToWelshDataset(val_ds_raw, Tokenizer_src, Tokenizer_trg, config['en'], config['cy'], config['seq_len'])

    max_len_src = 0
    max_len_trg = 0

    for item in ds_raw:
        src_ids = Tokenizer_src.encode(item['translation']['en']).ids
        trg_ids = Tokenizer_src.encode(item['translation']['cy']).ids
        max_len_src = max(max_len_src, len(src))
        max_len_trg = max(max_len_trg, len(trg))

    print(f'Maximum length of source sentence: {max_len_src}')
    print(f'Maximum length of target sentence: {max_len_trg}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, Tokenizer_src, Tokenizer_trg


    def get_model(config, vocab_src_len, vocab_trg_len):
        model = nn.Transformer(vocab_src_len, vocab_trg_len, config['seq_len'], config['seq_len'], config['d_model'])
        return model


def train_model(config):
# definw the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, Tokenizer_src, Tokenizer_trg = get_ds(config)
    model = get_model(config, len(Tokenizer_src.get_vocab()), len(Tokenizer_trg.get_vocab().to(device)))

    # Tensorboard
    writer = SummaryWriter(f'runs/{config["experiment_name"]}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preloaded_model']:
        model_filename = get_weights_file_name(config, config['preloaded_model'])
        print(f'Loading model from {model_filename}')
        state -= torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])



    loss_fn = nn.CrossEntropyLoss(ignore_index=Tokenizer_src.token_to_id(['[PAD]']), label_smoothing=0.1).to(device) #ignore the padding token, label smoothing is a regularization technique to prevent overfitting

    for epoch in range(initial_epoch, config['num_epochs']):


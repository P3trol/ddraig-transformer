from pathlib import Path

def get_config():
    return {
        'batch_size': 8, #number of sentences to process in one go
        'num_epochs': 20, #number of times to go through the dataset
        'lr': 10**-4,   #learning rate
        'seq_len': 128, #maximum length of a sentence
        'lang_src': 'ca', #language pair
        'lang_trg': 'de', #language pair
        'model_folder': 'weights', #path to save the model
        'model_name': 'tmodel_', #name of the model
        'preloaded_model': None, #path to a preloaded model
        'tokenizer_path': 'tokenizer_{0}.json', #path to the tokenizer
        'experiment_name': 'experiment', #name of the experiment
    }

def get_weights_file_name(config,_epoch: str):
    model_folder = config['model_folder']
    model_name = config['model_name']
    model_filename = f"{experiment_name}{epoch}.pt"
    return os.path.join(model_folder, model_filename)

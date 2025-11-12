import torch

def texts_to_tensor(samples, seq_len):
    samples_len = len(samples)
    inputs = torch.zeros((1, samples_len - 1, seq_len), dtype=torch.long)
    targets = torch.zeros((1, samples_len - 1, seq_len), dtype=torch.long)

    for sample_i in range(len(samples) - 1):
        for i in range(seq_len):
            inputs[0, sample_i, i] = samples[sample_i][i]
            targets[0, sample_i, i] = samples[sample_i + 1][i]

    return inputs, targets

def get_device(device = None):
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.4, 0)
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(0.4)
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)
    
def clean_df(df):
    df = df[df['text'].str.len() >= 10]
    df['name_ru'] = df['name_ru'].fillna('нет')
    return df
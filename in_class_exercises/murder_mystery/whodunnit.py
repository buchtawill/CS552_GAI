# Parts of this lab are based on https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html

# TODO: To get started, download train.jw and train.en from Canvas
# and upload them to your Colab file system.
import math
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
import io

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
PAD_IDX = tokenizer.pad_token_id
BOS_IDX = tokenizer.bos_token_id
EOS_IDX = tokenizer.eos_token_id
BATCH_SIZE = 128

if 'train_data' not in globals().keys():
    def data_process(filepaths):
      raw_jw_iter = iter(io.open(filepaths[0], encoding="utf8"))
      raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
      data = []
      for (raw_jw, raw_en) in zip(raw_jw_iter, raw_en_iter):
        jw_tensor_ = torch.tensor(tokenizer(raw_jw)['input_ids'], dtype=torch.long)
        en_tensor_ = torch.tensor(tokenizer(raw_en)['input_ids'], dtype=torch.long)
        data.append((jw_tensor_, en_tensor_))
      return data

    global train_data
    train_data = data_process(("train.jw", "train.en"))

def generate_batch(data_batch):
    jw_batch, en_batch = [], []
    for (jw_item, en_item) in data_batch:
        jw_batch.append(torch.cat([torch.tensor([BOS_IDX]), jw_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    jw_batch = pad_sequence(jw_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return jw_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch, num_workers=8)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # Dimensions represent 
        # Part (1a)
        # pe = pe.unsqueeze(1)  # TODO: check if the unsqueeze dimension is correct!
        pe = pe.unsqueeze(1)  # Yes dimension is correct. X is (t words, N batch size, num features (512))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Part (1b)
        # TODO: add positional embeddings to the inputs. Which of the lines below is correct?
        # ---> i.e. make sure the dimension correctly matches the unsqueezed dimension
        # ---> so the second line is correct
        # return x + self.pe[:x.size(0), :, :]
        return x + self.pe[:, :x.size(1), :]
        pass

class MT(nn.Module):
    def __init__(self, numTokens: int, padTokenId: int, device: torch.device):
        super().__init__()
        self.transformer = nn.Transformer(num_encoder_layers=5, num_decoder_layers=5)
        self.embedding = nn.Embedding(numTokens, self.transformer.d_model)
        self.classifier = nn.Linear(self.transformer.d_model, numTokens)
        # Part (2)
        # what should embedding size be for the PositionalEncoding?
        # Answer --> it needs to match the embedding size of the transformer
        self.pos_encoder = PositionalEncoding(self.transformer.d_model) 
        self.device = device
        self.padTokenId = padTokenId

    def forward(self, src: Tensor, trg: Tensor, teacher_forcing: int = 0) -> Tensor:
        # Part (3)
        # TODO: Embed all input tokens. Add positional embeddings to both the
        # encoder's and decoder's inputs. Then pass to Transformer. Make sure
        # the transformer doesn't try to cheat by looking into the future!
        # Also make sure not to omit padding tokens.
        # ...
        src_embedded = self.embedding(src) 
        trg_embedded = self.embedding(trg)
        src_embedded += self.pos_encoder(src_embedded)
        trg_embedded += self.pos_encoder(trg_embedded)
        
        X = src_embedded
        Y = trg_embedded

        mask = nn.Transformer.generate_square_subsequent_mask(Y.shape[0], device=self.device)  # causal mask
        out = self.classifier(self.transformer(X, Y,
                               tgt_mask=mask,
                               src_key_padding_mask=(src == PAD_IDX).transpose(0, 1),
                               memory_key_padding_mask=(src == PAD_IDX).transpose(0, 1),
                               tgt_key_padding_mask=(trg == PAD_IDX).transpose(0, 1)))
        return out
    
    def generate(self, src_tensor: Tensor, bos: int, eos: int, vocabLen: int) -> Tensor:
        model.eval()

        # Start out with just BOS.
        generated = torch.tensor([bos]).to(self.device).long().unsqueeze(1)

        # TODO: auto-regress until an EOS token is predicted
        max_length = 100
        for _ in range(max_length):
            output = model(src_tensor, generated)
            logits = output[-1, 0, :]  # Get the last token's logits
            # Part (4)
            # Get the most likely next token
            next_token = torch.argmax(logits, dim=-1)
            if next_token == eos:
                break
            generated = torch.cat((generated, next_token.unsqueeze(0).unsqueeze(0)), dim=0)
        generated_tokens = generated.squeeze(1).tolist()  # Remove batch dimension
        return generated_tokens

model = MT(len(tokenizer), PAD_IDX, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module):
    model.train()
    epoch_loss = 0
    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)

        # Part (5)
        # Apply the correct loss to the predictions
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)
        loss = criterion(output, trg)


        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Train MT model
N_EPOCHS = 10
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

# Part (6)
# TODO: Translate the magic sentence!
sentence = "Biod gammuc cohb bih poxf hba caidoxn cukix ndaax s."  # Magic sentence
inputs = torch.tensor(tokenizer(sentence)['input_ids']).unsqueeze(1).to(device)
result = model.generate(inputs, BOS_IDX, EOS_IDX, len(tokenizer))
trans_sentence = tokenizer.decode(result, skip_special_tokens=True)
print(trans_sentence)
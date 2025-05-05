import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import torch.nn.init as init

##################################################
# VQVAE
##################################################
EMBEDDING_DIM = 64
NUM_IMAGE_EMBEDDINGS = 128

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=2, padding=0),
        nn.ReLU(inplace=True)
    )

def double_convT(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=0),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True)
    )

class VQVAE(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv1 = double_conv(1, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, EMBEDDING_DIM)
        self.convT3 = double_convT(EMBEDDING_DIM, 128)
        self.convT2 = double_convT(128, 64)
        self.convT1 = double_convT(64, 1)

        self.e = torch.empty(EMBEDDING_DIM, NUM_IMAGE_EMBEDDINGS, requires_grad=True)
        self.e = nn.Parameter(self.e)
        init.kaiming_uniform_(self.e)

    def conv (self, x):
        return (self.conv3(self.conv2(self.conv1(x))))

    def convT (self, x):
        return self.convT1(self.convT2(self.convT3(((x)))))

    def forward (self, x):
        beforeBottleneck, idxs = self.encode(x)

        # This statement does not backprop through convT(...) to E
        reconstructions, afterBottleneck = self.decode(idxs)
        return reconstructions, idxs, beforeBottleneck, afterBottleneck

    def decode (self, idxs, beforeBottleneck = None):
        afterBottleneck = torch.swapaxes(self.e[:,idxs], 0, 1)
        if beforeBottleneck != None:  # training
            return self.convT(beforeBottleneck + (afterBottleneck - beforeBottleneck).detach()), afterBottleneck
        else:  # inference
            return self.convT(afterBottleneck), afterBottleneck

    def encode (self, x):
        beforeBottleneck = self.conv(x)
        dists = torch.norm(beforeBottleneck.unsqueeze(2) - self.e.unsqueeze(0).unsqueeze(3).unsqueeze(4), dim=1)
        idxs = torch.argmin(dists, dim=1)
        return beforeBottleneck, idxs

##################################################
# Transformer decoder
##################################################

NUM_IMAGE_TOKENS = 9*26

# You probably won't need to change this (but can if you want)
class Tokenizer:
    def __init__ (self, labelsFilename, minTokenIdx):
        self.minTokenIdx = minTokenIdx
        lines = open(labelsFilename, "rt").readlines()
        text = ""
        maxLen = -1
        for line in lines:
            line = line.rstrip()
            if len(line) > maxLen:
                maxLen = len(line)
            text += line
        self.maxLen = maxLen
        self.idxToCharMap = list(np.unique(list(text)))
        self.charToIdxMap = { c:i for (i, c) in enumerate(self.idxToCharMap) }

    def tokenize (self, strings, paddingToken):
        results = []
        for string in strings:
            result = [ (self.minTokenIdx + self.charToIdxMap[c]) for c in string ]
            result += [paddingToken] * (self.maxLen - len(string) + 1)  # "+ 1" -- add a padding token at the end
            results.append(result)
        return results

    def untokenize (self, idxs, paddingToken):
        if paddingToken not in idxs:
            maxIdx = len(idxs)
        else:
            maxIdx = np.min(np.nonzero(idxs == paddingToken)[0])
        return "".join([ self.idxToCharMap[(idx - self.minTokenIdx)] for idx in idxs[0:maxIdx] ])

# This might work better than a sinusoidal encoder for 2-d image inputs.
# You shouldn't need to change this.
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(AbsolutePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model, requires_grad=True))
        init.kaiming_uniform_(self.pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :, :]

# Define the TransformerDecoder model
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocabSize, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        # TODO: implement me
        # You will need an embedding layer to handle both image and text tokens.
        self.embedding = nn.Embedding(vocabSize, d_model)
        # You will also need a final linear layer to predict the next token.
        self.output_linear = nn.Linear(d_model, vocabSize)
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers
        )
        self.pos_encoder = AbsolutePositionalEncoding(d_model)

    def forward(self, inputs, maskLen, paddingToken):
        # TODO: Apply embedding layer and add positional encodings
        embedded = self.embedding(inputs)
        embedded = self.pos_encoder(embedded.permute(1, 0, 2))  # Shape: (seq_len, batch_size, d_model)
        
        tgt_mask = torch.nn.modules.transformer._generate_square_subsequent_mask(maskLen).to(device)
        memory = torch.zeros(X.shape).to(device)  # 0 memory since this is decoder-only
        output = self.transformer_decoder(X, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=(inputs.t()==paddingToken).float())
        # TODO: Apply linear layer
        return output


def generate (transformer, startingSequence, paddingToken):
    generated_tokens = []
    # TODO: implement me
    return generated_tokens

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae = VQVAE()
    vqvae.load_state_dict(torch.load("./vqvae_equations.cpt"))
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    # Load equation labels and initialize tokenizer
    tokenizer = Tokenizer("equation_labels.txt", NUM_IMAGE_EMBEDDINGS)
    sequenceLen = tokenizer.maxLen + NUM_IMAGE_TOKENS
    paddingToken = NUM_IMAGE_EMBEDDINGS + len(tokenizer.idxToCharMap)
    vocabSize = NUM_IMAGE_EMBEDDINGS + len(tokenizer.idxToCharMap) + 1  #  "+ 1" for padding
    paddingToken = vocabSize - 1

    # Load pre-computed image token indices
    allIdxs = np.load("allIdxs.npz")['allIdxs'] # (30000, 268)
    print(allIdxs[0])
    print()
    print(allIdxs[1])
    exit()

    # Initialize input_data and target_data from allIdxs
    # Apply left-shift for next-token prediction
    input_data = torch.tensor(allIdxs[:, :-1], dtype=torch.long)
    target_data = torch.tensor(allIdxs[:, 1:], dtype=torch.long)

    # Create DataLoader for training
    BATCH_SIZE = 32
    train_dataset = TensorDataset(input_data, target_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the model (using a recommended architecture)
    d_model = 256
    nhead = 8
    num_layers = 6
    dim_feedforward = 512
    transformer = TransformerDecoderModel(d_model, nhead, num_layers, dim_feedforward).to(device)  # Add any other params you want

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(transformer.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        transformer.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Transformer expects batch as second dimension
            inputs = inputs.t().to(device)
            targets = targets.t().to(device)
            outputs = transformer(inputs, sequenceLen, paddingToken)

            # TODO: compute loss and backprop

        # Every epoch, show an example of an image and also image2text translation
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        transformer.eval()
        gt = targets[NUM_IMAGE_TOKENS:,0].detach().cpu().numpy()
        print("ground-truth: ", tokenizer.untokenize(e, paddingToken))
        # TODO: use VQVAE decoder to render the image from image tokens
        # TODO: use Transformer decoder to perform image2text translation
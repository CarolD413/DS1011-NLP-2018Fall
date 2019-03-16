#!/usr/bin/env python
# coding: utf-8

################################################################################################
## Citation: Lab8 - http://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking ##
################################################################################################

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Preprocess Data

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

PAD_TAG = "<pad>"
SOS_TAG = "<sos>"
EOS_TAG = "<eos>"
UNK_TAG = "<unk>"

MAX_LEN = 200

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: PAD_TAG, 1: SOS_TAG,2:EOS_TAG, 3:UNK_TAG}
        self.n_words = 4  # Count PAD, SOS, EOS, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def removeSpace(s):
    s = re.sub(' +',' ', s).strip()
    return s

def readLangs(lang1, lang2, datasettype, reverse=False):
    print("Reading lines...")

    filename1 = '%s.tok.%s' % (datasettype, lang1)
    filename2 = '%s.tok.%s' % (datasettype, lang2)
        
    # Read the file and split into lines
    lines_1 = open(filename1, encoding='utf-8').        read().strip().split('\n')
    lines_2 = open(filename2, encoding='utf-8').        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(lines_1[i]),lines_2[i]] for i in range(len(lines_1))]
    print('Pair1:', pairs[1])
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LEN and         len(p[1].split(' ')) < MAX_LEN 
#     and \
#         p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, datasettype, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, datasettype, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('en', 'vi', 'train', True)
print(random.choice(pairs))


# # Dataset

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word]  
            if word in lang.word2index else 3 for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device) ##
    
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1]) 
    return (input_tensor, torch.cat((torch.tensor([SOS_TOKEN], dtype=torch.long, device=device),target_tensor)))
    

class Dataset(Dataset):
    def __init__(self,datasettype):
        input_lang, output_lang, pairs = prepareData('en', 'vi',datasettype, True) 
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        src = tensorsFromPair(self.pairs[index])[0]
        trg = tensorsFromPair(self.pairs[index])[1]
        return src, trg

def collate_fn(data):
    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).to(device)
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.cuda.LongTensor(seq[:end])
        return padded_seqs, lens

    data.sort(key=lambda x: len(x[0]), reverse=True) #sort according to length of src seqs
    src_seqs, trg_seqs = zip(*data)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    trg_seqs, trg_lens = _pad_sequences(trg_seqs)

    #(batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0,1)
    trg_seqs = trg_seqs.transpose(0,1)

    return src_seqs, src_lens, trg_seqs, trg_lens


# ### verify dataset


BATCH_SIZE = 16
val_dataset = Dataset('dev')
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=collate_fn,
                                           shuffle=True)

print("Number of source-target pairs:", len(val_dataset))
print("Input language: "+ val_dataset.input_lang.name + '('+str(val_dataset.input_lang.n_words)+')')
print("Output language: "+ val_dataset.output_lang.name + '('+str(val_dataset.output_lang.n_words)+')')


train_data = Dataset('train')
# Configure models
attn_model = 'dot'
hidden_size = 256
embed_size = 256
n_layers = 1
dropout = 0.1
batch_size = 16
checkpoint_dir = "checkpoints"

# Configure training/optimization
clip = 50
learning_rate = 0.001
decoder_learning_ratio = 5.0
n_epochs = 20


# ## Attention


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))              / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x.long()) * math.sqrt(self.d_model)


import math, copy, time
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SelfAttnEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(SelfAttnEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features, device=device))
        self.b_2 = nn.Parameter(torch.zeros(features, device=device))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class SelfAttnDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(SelfAttnDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
    
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


from torch.autograd import Variable
from sacrebleu import raw_corpus_bleu
def evaluate(model, test_loader, k=1, max_length=None):
    output = []
    h_t = []
    p = []
    score = 0
    count = 0
    for batch_idx, batch in enumerate(test_loader):
        src_batch, src_lens, trg_batch, trg_lens = batch
        batch_size = src_batch.shape[1]
        src_batch = src_batch.transpose(0,1)
        with torch.no_grad():
            decoded_sentences = []
            decoded_words = []
            for b in range(batch_size):
                count += 1
                max_length = src_lens[b]
                trg_sentence = [output_lang.index2word[int(token)] for token in trg_batch[1:,b] if token != PAD_TOKEN]
                src = src_batch[b,:src_lens[b]].unsqueeze(0)
                src_mask = (src != PAD_TOKEN).unsqueeze(-2)

                memory = model.encode(src, src_mask)
                ys = torch.ones((1, 1),device = device).fill_(SOS_TOKEN).type_as(src.data)
                priors = [[ys ,0, 0]]
                sent_cand = ['' for i in range(k)]
                for di in range(2*max_length):
                    curr = {}
                    possible = []
                    for prior_data in priors:
                        ys, v, source_idx = prior_data
                        out = model.decode(memory, src_mask, 
                                           Variable(ys), 
                                           Variable(subsequent_mask(ys.size(1))
                                                    .type_as(src.data)))
                        prob = model.generator(out)
                        topv, topi = prob.data.topk(k)
                        for i in range(k):
                            possible.append(int(topi[:,i].squeeze().detach()))
                            curr[topv[0,i]+v] = [topi[:,i], topv[0,i]+v, source_idx]
                    sorted_v = sorted(curr.keys(),reverse=True)
                    top_k = sorted_v[:k]
                    temp = [x for x in sent_cand]
                    for i, index in enumerate(top_k):
                        token = int(curr[index][0])
                        source_idx = curr[index][-1]
                        curr[index][-1] = i
                        if token == EOS_TOKEN:
                            sent_cand[i] = temp[source_idx] + '<eos>'
                            break
                        else:
                            sent_cand[i] = temp[source_idx] + (output_lang.index2word[token] + " " )                    
                    if EOS_TOKEN == possible[0]:
                        decoded_words = sent_cand[0]
                        break
                    priors = [curr[index] for index in top_k]
                if not decoded_words:
                    decoded_words = '<eos>'
                trg_sentence = ' '.join(trg_sentence)
                print(decoded_words)
                s = raw_corpus_bleu(decoded_words,trg_sentence).score
                score += s
    print(count)
    return score/count


def save_checkpoint(model, checkpoint_dir):
    filename = "{}/selfAttn-{}.pth".format(checkpoint_dir, time.strftime("%d%m%y-%H%M%S"))
    torch.save(model.state_dict(), filename)
    print("Model saved.")
    
def train_step(src_batch, src_lens, trg_batch, trg_lens, model, optimizer, criterion):
    # Zero gradients of both optimizers
    optimizer.zero_grad()
    loss, em_accuracy, edit_distance = 0.0, 0.0, 0.0
    # Run words through encoder
    max_trg_len = max(trg_lens)
    batch_size = src_batch.shape[1]
    src_batch = src_batch.transpose(0,1)
    trg_batch = trg_batch.transpose(0,1)
    trg = trg_batch[:, :-1]
    trg_y = trg_batch[:,1:]
    src_mask  = (src_batch != PAD_TOKEN).unsqueeze(-2)
    trg_mask = make_std_mask(trg, PAD_TOKEN)
    loss_mask = (trg_y != PAD_TOKEN).float()
    out = model.generator(model.forward(src_batch, trg, src_mask, trg_mask))
    for di in range(max_trg_len-1):
        loss += (criterion(out[:,di,:], trg_y[:,di].long()) * loss_mask[:,di]).mean()
#     loss = loss / (sum(trg_lens)/len(trg_lens))
    loss = loss/max_trg_len
    loss.backward()
    model_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    
    return loss.item(), em_accuracy, edit_distance #, enc_grads, dec_grads        


def train(dataset, batch_size, n_epochs, model, optimizer, criterion, 
          checkpoint_dir=None, save_every=3000):
    train_iter = DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=collate_fn,
                                           shuffle=True)
    for i in range(n_epochs):
        tick = time.process_time()
        print("Epoch {}/{}".format(i+1, n_epochs))
        losses, accs, eds = [], [], []
        
        for batch_idx, batch in enumerate(train_iter):
            input_batch, input_lengths, target_batch, target_lengths = batch
            loss, accuracy, edit_distance = train_step(input_batch, input_lengths, target_batch, 
                                                        target_lengths, model, optimizer, criterion)
            
            losses.append(loss)
            accs.append(accuracy)
            eds.append(edit_distance)
            if batch_idx % 100 == 0 and batch_idx != 0:
                print("batch: {}, loss: {}, accuracy: {}, edit distance: {}".format(batch_idx, loss, accuracy, 
                                                                                   edit_distance))
            
        tock = time.process_time()
        print("Time: {} Avg loss: {} Avg acc: {} Edit Dist.: {}".format(
            tock-tick, np.mean(losses), np.mean(accs), np.mean(eds)))
        save_checkpoint(model, checkpoint_dir)
        print(evaluate(model, val_loader))


# Initialize models
c = copy.deepcopy
h = 8
d_model = 256
d_ff = 512
dropout = 0.1
src_vocab = input_lang.n_words
tgt_vocab = output_lang.n_words
N = 6
learning_rate = 0.001


attn = MultiHeadedAttention(h, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
position = PositionalEncoding(d_model, dropout)
model = EncoderDecoder(
        SelfAttnEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        SelfAttnDecoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)).to(device)
model.train()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
criterion = nn.NLLLoss(ignore_index = PAD_TOKEN, reduction = 'none')


train(train_data, batch_size, n_epochs, model, optimizer, criterion, checkpoint_dir)

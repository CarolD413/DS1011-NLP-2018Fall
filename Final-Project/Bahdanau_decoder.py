#!/usr/bin/env python
# coding: utf-8

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
    return len(p[0].split(' ')) < MAX_LEN and len(p[1].split(' ')) < MAX_LEN 
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
    return (input_tensor, target_tensor)


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
hidden_size = 300
embed_size = 300
n_layers = 1
dropout = 0.1
batch_size = 16
checkpoint_dir = "checkpoints"

# Configure training/optimization
clip = 50
learning_rate = 0.001
decoder_learning_ratio = 5.0
n_epochs = 50


# # The Seq2Seq Model

# ## Attention

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.method = 'Bahdanau'
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        elif self.method == 'Bahdanau':
            self.W = nn.Linear(self.hidden_size, hidden_size)
            self.U = nn.Linear(self.hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(1, hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        
    def forward(self, last_hidden, encoder_outputs, src_len=None):
        
        # Create variable to store attention energies
        length = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        score = self.cal_score(last_hidden , encoder_outputs)
        if src_len is not None:
            mask = []
            for b in range(batch_size):
                mask.append([0] * src_len[b] + [1] * (encoder_outputs.size(0) - src_len[b]))
            mask = (torch.cuda.ByteTensor(mask).unsqueeze(1)) # [B,1,T]
            score = score.masked_fill(mask, -1e9)
        attn_weights = F.softmax(score, dim = 2)
        context_vector = torch.bmm(attn_weights, encoder_outputs.transpose(0,1))
        return context_vector, attn_weights
    
    def cal_score(self, hidden, encoder_output):
        length = encoder_output.shape[0]
        batch_size = encoder_output.shape[1]
        last_hidden = hidden.repeat(length,1,1)
        if self.method == 'dot':
            return None

        elif self.method == 'general':
            print(last_hidden.shape)
            print(self.attn(encoder_output).shape)
            return energy

        elif self.method == 'concat':
            energy =  torch.tanh(self.attn(torch.cat([last_hidden, encoder_output],dim=2)))
            energy = energy.transpose(0, 1).transpose(1,2)
            v = self.v.repeat(batch_size,1 , 1)
            return torch.bmm(v, energy)
        
        elif self.method == 'Bahdanau':
            energy = torch.tanh(self.W(last_hidden)+self.U(encoder_output))
            energy = energy.transpose(0, 1).transpose(1,2)
            v = self.v.repeat(batch_size,1 , 1)
            return torch.bmm(v, energy)


# The Encoder
# -------
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.

class EncoderRNN(nn.Module):
    def __init__(self, input_size,embed_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=PAD_TOKEN)
#         self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
    def forward(self, input_seqs, input_lengths, hidden=None, cell = None):
        embedded = self.embedding(input_seqs.long()) #input_seq: T*B, embedded: T*B*H
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
#         outputs, hidden = self.gru(packed, hidden)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# The Decoder
# ------------------
# decoder is also a RNN

# ### 1. Decoder w/o Attention


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layer=1, dropout=0,attention=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = dropout
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.softmax = nn.LogSoftmax(dim=1)
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=PAD_TOKEN)
        self.embedding_dropout = nn.Dropout(dropout)
#         self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
#         self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=False)
        self.lstm = nn.LSTM(embedding_size+hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=False)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
#         self.out = nn.Linear(hidden_size * 2, output_size)
        self.out = nn.Linear(hidden_size, output_size)
        if attention:
            self.attn = Attn(hidden_size)
# Bahdanau
        
    def forward(self, input_seq, last_hidden, src_len = None , encoder_outputs=None):
        batch_size = input_seq.size(0)
        input_seq = input_seq.long()
        embedded = self.embedding(input_seq).view(1, input_seq.size(0), -1)
        embedded = self.embedding_dropout(embedded)
        h,c = last_hidden
        context_vector, attn_weights = self.attn(h, encoder_outputs, src_len)
        rnn_input = torch.cat((embedded, context_vector.transpose(0,1)), 2)
        rnn_output, hidden = self.lstm(rnn_input, last_hidden)
        rnn_output = rnn_output.squeeze(0)
        output = F.log_softmax(self.out(rnn_output), dim = 1)
        return output, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



from torch.autograd import Variable
from sacrebleu import corpus_bleu
def evaluate(encoder, decoder, test_loader, k=1, max_length=None):
    output = []
    h_t = []
    p = []
    score = 0
    count = 0
    for batch_idx, batch in enumerate(test_loader):
        src_batch, src_lens, trg_batch, trg_lens = batch
        batch_size = src_batch.shape[1]
#         pos_index = Variable(torch.LongTensor(range(batch_size)) * k).view(-1, 1)
        with torch.no_grad():
            decoded_sentences = []
            for b in range(batch_size):
                count += 1
                max_length = src_lens[b]
                trg_sentence = [output_lang.index2word[int(token)] for token in trg_batch[:,b] if token != PAD_TOKEN]
                encoder_outputs, encoder_hidden = encoder(src_batch[:src_lens[b],b].unsqueeze(1),
                                                          torch.LongTensor([src_lens[b]]))
                decoder_input = torch.LongTensor([[SOS_TOKEN]]).to(device)
                decoder_hidden = encoder_hidden[:decoder.n_layers*2]
                max_trg_len = trg_lens[b]
                decoded_words = []
                decoder_attentions = torch.zeros(batch_size, max_length, max_length)
                priors = [[decoder_input, decoder_hidden, encoder_outputs,decoder_attentions,0, 0]]
                sent_cand = ['' for i in range(k)]
                for di in range(2 * max_length):
                    curr = {}
                    possible = []
                    for prior_data in priors:
                        decoder_input, decoder_hidden, encoder_outputs, decoder_attentions, v, source_idx = prior_data
                        decoder_output, decoder_hidden, decoder_attention = decoder(
                            decoder_input, decoder_hidden, [src_lens[b]], encoder_outputs)
                        topv, topi = decoder_output.data.topk(k)
#                         decoder_attentions[di] = decoder_attention.data
                        for i in range(k):
                            possible.append(int(topi[:,i].squeeze().detach()))
                            curr[topv[0,i]+v] = [topi[:,i], decoder_hidden, encoder_outputs, 
                                                 decoder_attentions, topv[0,i]+v, source_idx]

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
                s = corpus_bleu(decoded_words,trg_sentence).score
                score += s
        
    print(count)
    return score/count


def save_checkpoint(encoder, decoder, checkpoint_dir):
    enc_filename = "{}/enc-{}.pth".format(checkpoint_dir, time.strftime("%d%m%y-%H%M%S"))
    dec_filename = "{}/dec-{}.pth".format(checkpoint_dir, time.strftime("%d%m%y-%H%M%S"))
    torch.save(encoder.state_dict(), enc_filename)
    torch.save(decoder.state_dict(), dec_filename)
    print("Model saved.")
    

def train_step(src_batch, src_lens, trg_batch, trg_lens, encoder, decoder, 
               encoder_optimizer, decoder_optimizer, criterion):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss, em_accuracy, edit_distance = 0.0, 0.0, 0.0
    # Run words through encoder
    batch_size = src_batch.shape[1]
    encoder_outputs, encoder_hidden = encoder(src_batch, src_lens)
    # Prepare input and output variables
    trg_mask = trg_batch.to(device)
    trg_mask = (trg_mask != PAD_TOKEN).float()
#     trg_mask[trg_mask != PAD_TOKEN] = 1
    decoder_input = torch.LongTensor([SOS_TOKEN] * batch_size).to(device)
#     decoder_hidden = encoder_hidden[:decoder.n_layers*2] # Use last (forward) hidden state from encoder
    decoder_hidden = (torch.sum(encoder_hidden[0][:decoder.n_layers*2],dim = 0).unsqueeze(0), 
                      torch.sum(encoder_hidden[1][:decoder.n_layers*2],dim = 0).unsqueeze(0))
    max_trg_len = max(trg_lens)
    # Run through decoder one time step at a time using TEACHER FORCING=1.0
    TEACHER_FORCING = 1
    
#     decoder_output_ls = []
    for t in range(max_trg_len):
#         decoder_output, decoder_hidden, decoder_attn = decoder(
#             decoder_input, decoder_hidden, encoder_outputs
#         )
        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, 
                                                 decoder_hidden, src_lens, encoder_outputs)
        if TEACHER_FORCING:
            decoder_input = trg_batch[t]
        else:
            topv, topi = decoder_output.topk(1)
            decoder_input = topi  # detach from history as input
        loss += (criterion(decoder_output, trg_batch[t].long()) * trg_mask[t]).mean()
    loss = loss / max_trg_len
    loss.backward()
    
    # Clip gradient norms
    enc_grads = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dec_grads = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item(), em_accuracy, edit_distance #, enc_grads, dec_grads        

train_iter = DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           shuffle=True)

def train(dataset, batch_size, n_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
          checkpoint_dir=None, save_every=3000):
    global_step = 0
    
#     encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min')
#     decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min')
    for i in range(n_epochs):
        tick = time.process_time()
        print("Epoch {}/{}".format(i+1, n_epochs))
        
        losses, accs, eds = [], [], []
        
        for batch_idx, batch in enumerate(train_iter):
            global_step += 1
            input_batch, input_lengths, target_batch, target_lengths = batch
#             encoder.train()
#             decoder.train()
            loss, accuracy, edit_distance = train_step(input_batch, input_lengths, target_batch, target_lengths,
                                                       encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            
            losses.append(loss)
            accs.append(accuracy)
            eds.append(edit_distance)
            
            
            if batch_idx % 100 == 0 and batch_idx != 0:
                print("batch: {}, loss: {}, accuracy: {}, edit distance: {}".format(batch_idx, loss, accuracy, 
                                                                                   edit_distance))
#                 bleu = evaluate(encoder, decoder, val_loader)
#                 print(bleu)

        tock = time.process_time()
#         bleu = evaluate(encoder, decoder, val_loader)
#         print(bleu)
        print("Time: {} Avg loss: {} Avg acc: {}".format(
            tock-tick, np.mean(losses),  np.mean(accs)))
        save_checkpoint(encoder, decoder, checkpoint_dir)
        
        
        bleu = evaluate(encoder, decoder, val_loader)
                
#         encoder_scheduler.step(100-bleu)
#         decoder_scheduler.step(100-bleu)
        
    


# Initialize models

encoder = EncoderRNN(input_lang.n_words, embed_size, hidden_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, output_lang.n_words).to(device)

# Initialize optimizers and criterion
encoder.train()
decoder.train()
learning_rate = 0.001
encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss(ignore_index = PAD_TOKEN, reduction = 'none')

train(train_data, batch_size, n_epochs, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, criterion, checkpoint_dir)

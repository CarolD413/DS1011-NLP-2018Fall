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
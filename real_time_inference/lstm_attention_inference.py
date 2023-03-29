import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

encodings = torch.load('trained_data/encodings.dict')
word_idx = torch.load('trained_data/word_idx.dict')

encoder = torch.load('../../encoder.model', map_location=torch.device(device))
decoder = torch.load('../../decoder.model', map_location=torch.device(device))
encoder.eval()
decoder.eval()

def evaluate(frames, max_length = 64):
    global encoder
    global decoder
    
    with torch.no_grad():
        encoder_hidden = encoder.initHidden()

        encoder_output, encoder_hidden = encoder(frames, encoder_hidden)

        decoder_input = torch.tensor([[encodings['SOS']]], device=device)  # Start of sentence

        decoder_hidden = encoder_hidden

        decoded_words = ''
        decoder_attentions = torch.zeros(max_length, max_length, device=device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden[0], encoder_output)

            decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)

            if topi.item() == encodings['EOS']:
                decoded_words += '.'
                break
            else:
                decoded_words += word_idx[topi.item()] + ' '

            decoder_input = topi.detach()

        return decoded_words, decoder_attentions[:di + 1]
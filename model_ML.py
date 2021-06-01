!pip -V
!pip install docx2txt
!pip install tensorflow==2.4.0

from google.colab import drive
drive.mount('/content/drive')

# Library needed for building and deploying the model
import re
import os
import numpy as np
import docx2txt
import pickle
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import optimizers, metrics, backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def seq2seq(hidden_size, input_chars, target_chars):

    # main model
    encoder_lstm = LSTM(hidden_size, recurrent_dropout=0.3, return_sequences=True, return_state=False, name='encoderLSTM1')
    encoder_out = encoder_lstm(Input(shape=(None, input_chars), name='data_encoder'))
    encoder_lstm = LSTM(hidden_size, recurrent_dropout=0.3, return_sequences=False, return_state=True, name='encoderLSTM2')
    encoder_out, state_h, state_c = encoder_lstm(encoder_out)
    encoder_states = [state_h, state_c]

    # initial state is 'encoder_states'.
    decoder_lstm = LSTM(hidden_size, dropout=0.3, return_sequences=True, return_state=True, name='decoderLSTM')
    decoder_out, _, _ = decoder_lstm(Input(shape=(None, target_chars), name='data_decoder'), initial_state=encoder_states)
    decoder_dense = Dense(target_chars, activation='softmax', name='decoder_dense')
    decoder_out = decoder_dense(decoder_out)
    
    # Define model
    model = Model(inputs=[encoder_in, decoder_in], outputs=decoder_out)
    optimizer = optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define encoder model separately.
    encoder_model = Model(inputs=encoder_in, outputs=encoder_states)

    # Define decoder model separately.
    decoder_inputs_states = [Input(shape=(hidden_size,)), Input(shape=(hidden_size,))]
    decoder_out, state_h, state_c = decoder_lstm(decoder_in, initial_state=decoder_inputs_states)
    decoder_states = [state_h, state_c]
    decoder_model = Model(inputs=[decoder_in] + decoder_states_inputs, outputs=[decoder_dense(decoder_out)] + decoder_states)

    return model, encoder_model, decoder_model

np.random.seed(0)

SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.
CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ') # list of alphabet
cleaned_chars = '[#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'  # character must be clean


class CharacterTable(object):

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.size = len(self.chars)
        self.char2index = dict((c, i) for i, c in enumerate(self.chars))
        self.index2char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, encoded, row):
        # One-hot encode given string encoded
        encoding = np.zeros((row, len(self.chars)), dtype=np.float32)
        for i, c in enumerate(encoded):
            encoding[i, self.char2index[c]] = 1.0
        return encoding

    def decode(self, decoded, argmax=True):
        # Decode the given vector or 2D array to their character output.
        if argmax:
            idx = decoded.argmax(axis=-1)
        else:
            idx = decoded
        char = ''
        chars = char.join(self.index2char[i] for i in idx)
        return idx, chars

    def sample_multinomial(self, preds, temp=1.0):
        # Reshaped to 1D array of shape
        preds_reshape = np.reshape(preds, len(self.chars)).astype(np.float64)
        index = np.argmax(np.random.multinomial(1, (np.exp(np.log(preds_reshape) / temp) / np.sum(np.exp(np.log(preds_reshape) / temp))), 1))
        char  = self.index2char[index]
        return index, char

def tokenize(text):
    tokens = [re.sub(cleaned_chars, '', token) for token in re.split("[-\n ]", text)]
    return tokens

def read_text(data_path, list_of_books):
    for book in list_of_books:
        file_path = os.path.join(data_path, book)
        text = docx2txt.process(file_path)
    return text

def add_spelling_error(token, error_rate):
    # What to do with the token when countering spelling error
    assert(0. <= error_rate < 1.)
    if len(token)<3:
        return token
    rand = np.random.rand()
    random_char = np.random.randint(len(token)) # From first to third quadran
    probability = error_rate / 4. # From paper, in which 4 different spelling error could occur
    if rand<probability:
        token = token[:random_char] + np.random.choice(CHARS) + token[random_char + 1:] #tokenize it
    elif probability < rand < probability * 2:
        token = token[:random_char] + token[random_char + 1:]
    elif probability * 2 < rand < probability * 3:
        token = token[:random_char] + np.random.choice(CHARS) + token[random_char:]
    elif probability * 3 < rand < probability * 4:
        random_char_last = np.random.randint(len(token) - 1)
        token = token[:random_char_last]  + token[random_char_last + 1] + token[random_char_last] + token[random_char_last + 2:]
    else:
        pass
    return token

def transform(tokens, maxlen, error_rate=0.3, shuffle=True):
    #Transform tokens into model inputs and targets, also padded to maxlen with EOS character.
    if shuffle:
        np.random.shuffle(tokens)
    encoderTokens = []
    decoderTokens = []
    targetTokens = []
    for token in tokens:
        encoder = add_speling_erors(token, error_rate=error_rate)
        # Padded to maxlen
        encoder += EOS * (maxlen - len(encoder))
        encoderTokens.append(encoder)
    
        decoder = SOS + token
        decoder += EOS * (maxlen - len(decoder))
        decoderTokens.append(decoder)
    
        target = decoder[1:]
        target += EOS * (maxlen - len(target))
        targetTokens.append(target)
        
        assert(len(encoder) == len(decoder) == len(target))
    return encoder_tokens, decoder_tokens, target_tokens

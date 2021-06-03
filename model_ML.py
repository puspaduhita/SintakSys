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
    #function to tokenize the text
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
        
    return encoder_tokens, decoder_tokens, target_tokens

def load_data(encoder_loop, decoder_loop, target_loop):
    """Utility function to load data into required model format."""
    inputs = zip(encoder_loop, decoder_loop)
    while(True):
        encoder_input, decoder_input = next(inputs)
        target = next(target_loop)
        yield ([encoder_input, decoder_input], target)

def batch(tokens, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(tokens, reverse):
        while True:
            for token in tokens:
                if reverse:
                    token = token[::-1]
                yield token
            
    token_iter = generate(tokens, reverse)
    data_padded = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iter)
            data_padded[i] = ctable.encode(token, maxlen)
        yield data_padded

def predict(inputs, targets, input_ctable, target_ctable, maxlen, reverse, encoder_model, decoder_model,
                     len_test, sample_mode='argmax', random=True):
    """Function to process (predict) the dataset, hence this function will return the decoded tokens 
    added with input and target tokens for manual comparing purposes"""
    input_tokens = []
    target_tokens = []
    #Sign the index
    if random:
        index = np.random.randint(0, len(inputs), len_test)
    else:
        index = range(len_test)
    #Make list for the index    
    for idx in index:
        input_tokens.append(inputs[idx])
        target_tokens.append(targets[idx])

    input_sequences = batch(input_tokens, maxlen, input_ctable, len_test, reverse)
    states_value = encoder_model.predict(next(input_sequences))
    
    target_sequences = np.zeros((len_test, 1, target_ctable.size))
    target_sequences[:, 0, target_ctable.char2index[SOS]] = 1.0

    decoded_tokens = [''] * len_test
    for _ in range(maxlen):
        char_error, h, c = decoder_model.predict([target_sequences] + states_value)

        # Reset the target sequences.
        target_sequences = np.zeros((len_test, 1, target_ctable.size))

        # Sample next character using argmax or multinomial mode.
        sampled_chars = []
        for i in range(len_test):
            if sample_mode == 'argmax':
                next_index, next_char = target_ctable.decode(char_error[i], calc_argmax=True)
            elif sample_mode == 'multinomial':
                next_index, next_char = target_ctable.sample_multinomial(char_error[i], temperature=0.5)

            decoded_tokens[i] += next_char
            sampled_chars.append(next_char) 
            # Update target sequence with index of next character.
            target_sequences[i, 0, next_index] = 1.0

        stop_char = set(sampled_chars)
        if len(stop_char) == 1 and stop_char.pop() == EOS:
            break
            
        # Update states.
        states_value = [h, c]
    
    # Sampling finished.
    input_tokens   = [re.sub('[%s]' % EOS, '', token) for token in input_tokens]
    decoded_tokens = [re.sub('[%s]' % EOS, '', token) for token in decoded_tokens]
    target_tokens  = [re.sub('[%s]' % EOS, '', token) for token in target_tokens]
    return input_tokens, target_tokens, decoded_tokens


def restore_model(path_to_full_model, hidden_size):
    """Restore model to reconstruct the encoder and decoder."""
    model = load_model(path_to_full_model)
    # Using 2 LSTM
    encoder_lstm1 = model.get_layer('encoder_lstm_1')
    encoder_lstm2 = model.get_layer('encoder_lstm_2')
    
    encoder_outputs = encoder_lstm1(model.input[0])
    _, state_h, state_c = encoder_lstm2(encoder_outputs)
    encoder_model = Model(inputs=model.input[0], outputs=[state_h, state_c])

    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(model.input[1], initial_state=decoder_states_inputs)
    decoder_softmax = model.get_layer('decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_outputs)
    decoder_model = Model(inputs=[model.input[1]] + [Input(shape=(hidden_size,)), Input(shape=(hidden_size,))],
                          outputs=[decoder_outputs] + [state_h, state_c])
    return encoder_model, decoder_model

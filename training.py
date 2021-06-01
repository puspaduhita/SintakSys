import os
import numpy as np

from model_ML import CharacterTable
from model_ML import tokenize, read_text
from model_ML import transform, add_speling_erors
from model_ML import datagen, batch, decode_sequences

np.random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define constant parameter
error_rate = 0.85
train_batch_size = 64
val_batch_size = 64
hidden_size = 128
reverse = True
sample_mode = 'argmax'
num_epochs = 10

data_path = '/content/'
train_books = ['train fix.docx']
val_books = ['train.docx']


if __name__ == '__main__':
    # Prepare training data.
    text  = read_text(data_path, train_books)
    train_token = list(filter(None, set(tokenize(text))))

    maxlen = max([len(token) for token in train_token]) + 2 # plus SOS & EOS
    train_encoder, train_decoder, train_target = transform(train_token, maxlen, error_rate=error_rate, shuffle=False)

    input_chars = set(' '.join(train_encoder))
    nb_input_chars = len(input_chars)
    target_chars = set(' '.join(train_decoder))
    nb_target_chars = len(target_chars)
    
    # Prepare validation data.
    text = read_text(data_path, val_books)
    val_tokens = list(filter(None, tokenize(text)))

    val_maxlen = max([len(token) for token in val_tokens]) + 2 # plus SOS & EOS
    val_encoder, val_decoder, val_target = transform(val_tokens, maxlen, error_rate=error_rate, shuffle=False)
    
    train_steps = len(train_token) // train_batch_size
    val_steps = len(val_tokens) // val_batch_size

    # Define training and evaluation configuration.
    input_ctable  = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)

    # Compile the model.
    model, encoder_model, decoder_model = seq2seq(hidden_size, nb_input_chars, nb_target_chars)
    print(model.summary())

    # Train and evaluate.
    for epoch in range(num_epochs):

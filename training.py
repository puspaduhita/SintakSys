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
        print('Main Epoch {:d}/{:d}'.format(epoch + 1, nb_epochs))

        train_encoder, train_decoder, train_target = transform(vocab, maxlen, error_rate=error_rate, shuffle=True)   
        # Load the data for training and validation
        train_loader = load_data(batch(train_encoder, maxlen, input_ctable,train_batch_size, reverse),
                               batch(train_decoder, maxlen, target_ctable, train_batch_size), 
                               batch(train_target, maxlen, target_ctable, train_batch_size))
        val_loader = load_data(batch(val_encoder, maxlen, input_ctable, val_batch_size, reverse),
                             batch(val_decoder, maxlen, target_ctable, val_batch_size), 
                             batch(val_target, maxlen, target_ctable, val_batch_size))
        # Evaluate the model
        model.fit(train_loader,
                  steps_per_epoch=train_steps,
                  epochs=1, verbose=1,
                  validation_data=val_loader,
                  validation_steps=val_steps)

        # On epoch end - decode a batch of misspelled tokens from the
        # validation set to visualize speller performance.
        nb_tokens = 5
        input_tokens, target_tokens, decoded_tokens = predict(
            val_encoder, val_target, input_ctable, target_ctable,
            maxlen, reverse, encoder_model, decoder_model, nb_tokens,
            sample_mode=sample_mode, random=True)
        
        print('====================================')
        print('Input tokens:  ', input_tokens)
        print('Decoded tokens:', decoded_tokens)
        print('Target tokens: ', target_tokens)
        print('====================================')
        
        # Save the model at end of each epoch.
        model_file = '_'.join(['seq2seq', 'epoch', str(epoch + 1)]) + '.h5'
        model_dir = '/content'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_path = os.path.join(model_dir, model_file)
        print('Saving full model to {:s}'.format(save_path))
        model.save(save_path)

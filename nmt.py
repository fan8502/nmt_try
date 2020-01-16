from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io


import time

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    
    w = unicode_to_ascii(w.lower().strip())
    w = custom_word_tokenizer(w)
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

removable = "(\"|,|\$|\(|\)|\-|\--|\!|\")"

def custom_word_tokenizer(text):
    text = text.replace("\'s", " is")
    text = text.replace("they're", "they are")
    text = text.replace("you're", "you are")

    text= re.sub("I've","I have",text)
    text= re.sub("they've","they have",text)
    text= re.sub("we've","we have",text)
    
    text= re.sub("I'm","I am",text)
    text= re.sub("I'd","I would",text)
    text= re.sub("I'll","I will",text)
    
    text= re.sub("isn’t","is not",text)
    text= re.sub("hasn’t","has not",text)
    text= re.sub("(don’t)|(Don't)","do not",text)
    text= re.sub("(doesn’t)|(Doesn't)","does not",text)
    text= re.sub("(can’t)|(Can't)","can not",text)
    text= re.sub("couldn’t","could not",text)
    text= re.sub("aren’t","are not",text)
    text= re.sub("haven’t","have not",text)
    text= re.sub("(won’t)|(Won't)","will not",text)
    text= re.sub("wasn’t","was not",text)
    text= re.sub("(hadn’t)|(Hadn't)","had not",text)
    text= re.sub("(didn’t)|(Didn't)","did not",text)
    text= re.sub("(shouldn’t)|(Shouldn't)","should not",text)
    
    text= re.sub("she's","she is",text)
    text= re.sub("(He’s)|(he's)","he is",text)
    text= re.sub("(Let’s)|(let's)","let us",text)
    
    text= re.sub("(She’d)|(she'd)","She would",text)
    text= re.sub("(You’d)|(you'd)","You would",text)
    text= re.sub("(He’d)|(he'd)","He would",text)
    
    text= re.sub("(what’s)|(What's)","what is",text)
    text= re.sub("(where’s)|(Where's)","Where is",text)
    text= re.sub("-"," ",text)
    text= re.sub(removable,"",text)
    return text

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
    inp_lang, targ_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
          # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

          # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
        return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


if __name__ == '__main__':
	num_examples = 30000
	path_to_file = './data/nmt_input.txt'
	input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

	max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

	input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
	print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

	BUFFER_SIZE = len(input_tensor_train)
	BATCH_SIZE = 64
	steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
	embedding_dim = 256
	units = 1024
	vocab_inp_size = len(inp_lang.word_index)+1
	vocab_tar_size = len(targ_lang.word_index)+1

	dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
	dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
	example_input_batch, example_target_batch = next(iter(dataset))
	example_input_batch.shape, example_target_batch.shape

	encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

	# sample input
	sample_hidden = encoder.initialize_hidden_state()
	sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
	print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
	print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

	attention_layer = BahdanauAttention(10)
	attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

	print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
	print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

	decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

	sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

	print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

	optimizer = tf.keras.optimizers.Adam()
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

	EPOCHS = 10

	for epoch in range(EPOCHS):
	    start = time.time()

	    enc_hidden = encoder.initialize_hidden_state()
	    total_loss = 0

	    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
	        batch_loss = train_step(inp, targ, enc_hidden)
	        total_loss += batch_loss

	        if batch % 100 == 0:
	            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
	                                                   batch,
	                                                   batch_loss.numpy()))
	  # saving (checkpoint) the model every 2 epochs
	    if (epoch + 1) % 2 == 0:
	        checkpoint.save(file_prefix = checkpoint_prefix)

	    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
	                                      total_loss / steps_per_epoch))
	    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
	translate(u'The encoder output is calculated only once for one input.')
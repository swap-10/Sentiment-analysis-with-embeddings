import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_len = 100
trunc_mode = 'post'
pad_mode = 'post'
oov_tok = '<OOV>'
training_size = 20000

with open('.\sarcasm.json', 'r') as jf:
    datastore = json.load(jf)

headlines = []
labels = []

for item in datastore:
    headlines.append(item['headline'])
    labels.append(item['is_sarcastic'])

train_headlines = headlines[:training_size]
train_labels = labels[:training_size]
test_headlines = headlines[training_size:]
test_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_headlines)
train_sequences = tokenizer.texts_to_sequences(train_headlines)
train_padded = pad_sequences(train_sequences, maxlen=max_len, truncating=trunc_mode, padding=pad_mode)

test_sequences = tokenizer.texts_to_sequences(test_headlines)
test_padded = pad_sequences(test_sequences, maxlen=max_len, truncating=trunc_mode, padding=pad_mode)

train_padded = np.array(train_padded)
train_labels = np.array(train_labels)
test_padded = np.array(test_padded)
test_labels = np.array(test_labels)

# Model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# Training

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.97:
            print("Reached 97% accuracy so stopping training")
            self.model.stop_training=True

callback1 = myCallback()
num_epochs = 25
callbacks=[callback1]

history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), callbacks=callbacks, verbose=2)

def plot_metrics(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Num of Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()

plot_metrics(history, 'accuracy')
plot_metrics(history, 'loss')


# Utility mappings
reverse_map = dict([(value, key) for (key, value) in tokenizer.word_index.items()])
reverse_map[0] = ''

def idw(num):
    return reverse_map[num]

def idx2word(indices):
    return ' '.join([idw(num) for num in indices])

print(idx2word(train_padded[5]))
print(train_headlines[5])
print(train_labels[5])


# TSV Files

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

v_file = io.open('sarcasm_vec.tsv', 'w', encoding='utf-8')
m_file = io.open('sarcasm_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = idw(word_num)
    embeddings = weights[word_num]
    v_file.write('\t'.join([str(x) for x in embeddings]) + '\n')
    m_file.write(word + '\n')
v_file.close()
m_file.close()

predict_sentences = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
predict_sequences = tokenizer.texts_to_sequences(predict_sentences)
padded = pad_sequences(predict_sequences, maxlen=max_len, padding=pad_mode, truncating=trunc_mode)
print("The predictions are: \n", model.predict(padded))
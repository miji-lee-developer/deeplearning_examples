# Character Sequence RNN
import tensorflow as tf
import numpy as np

# sample = "if you want you"
sample = ("if you want to build a ship, don't drum up people together to \n"
          "collect wood and don't assign them tasks and work, but rather \n"
          "teach them to long for the endless immensity of the sea.")
idx2char = list(set(sample)) # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)} # char -> index

# hyper parameters
dic_size = len(char2idx) # RNN input size(one hot size)
hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) # final output size(RNN or softmax, etc.)
batch_size = 1 # one sample data, one batch
sequence_length = len(sample) - 1 # number of lstm rollings(unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample] # char to index
x_data = [sample_idx[:-1]] # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]] # Y data sample (1 ~ n) hello: ello

print('x_data: ', x_data)
x_ts = [idx2char[c] for c in x_data[0]]
print("\tx_data str: ", ''.join(x_ts))
print('y_data: ', y_data)
y_ts = [idx2char[c] for c in y_data[0]]
print("\ty_data str: ", ''.join(y_ts))

x_one_hot_eager = tf.one_hot(x_data, num_classes) # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
x_one_hot_numpy = tf.keras.utils.to_categorical(x_data, num_classes) # it'll generate numpy array, either way works
y_one_hot_eager = tf.one_hot(y_data, num_classes)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=num_classes, input_shape=(sequence_length, x_one_hot_eager.shape[2]), return_sequences=True))
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes, activation='softmax')))
tf.model.summary()
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
# tf.model.fit(x_one_hot_eager, y_one_hot_eager, steps_per_epoch=5000)
tf.model.fit(x_one_hot_numpy, y_one_hot_eager, epochs=50, steps_per_epoch=(len(x_one_hot_numpy)))

# predictions = tf.model.predict(x_one_hot_eager, steps=1)
predictions = tf.model.predict(x_one_hot_numpy, steps=1)

for i, prediction in enumerate(predictions):
    # print char using argmax, dict
    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print("\tPrediction str: \n", ''.join(result_str))
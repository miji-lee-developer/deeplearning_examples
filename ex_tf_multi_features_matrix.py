# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

x_data = np.array(x_data)
y_data = np.array(y_data)
print(x_data.shape, y_data.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.Input(shape=3))
tf.model.add(tf.keras.layers.Dense(1, activation='linear'))

# tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))   # input_dim=3 gives multi-variable regression
# tf.model.add(tf.keras.layers.Activation('linear'))  # this line can be omitted, as linear activation is default

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=100)

y_predict = tf.model.predict(np.array([[72., 93., 90.]]))
print(y_predict)
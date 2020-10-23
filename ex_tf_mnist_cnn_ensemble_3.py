import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# fit model on dataset
def fit_model(x_train, y_train):
    learning_rate = 0.001

    # define model
    model = tf.keras.models.Sequential()

    # L1
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(drop_rate))

    # L2
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(drop_rate))

    # L3
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(drop_rate))

    # L4 fully connected
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    return model


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

num_models = 10
models = []
histories = []
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

training_epochs = 50
batch_size = 64

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=False
)

datagen.fit(x_train)

for _ in range(num_models):
    model = fit_model(x_train, y_train)
    models.append(model)

for i in range(num_models):
    history = models[i].fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      epochs=training_epochs, steps_per_epoch=(len(x_train)//batch_size),
                                      verbose=0, callbacks=[annealer])
    histories.append(history)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}".format(i+1, training_epochs, max(history.history['accuracy'])))

results = np.zeros((x_test.shape[0], 10))

for i in range(num_models):
    results = results + models[i].predict(x_test)

results = np.argmax(results, axis=1)
print("results: ", results)
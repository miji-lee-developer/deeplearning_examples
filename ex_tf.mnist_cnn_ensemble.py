import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import random

# fit model on dataset
def fit_model(x_train, y_train):
    learning_rate = 0.001
    training_epochs = 12
    batch_size = 128
    # drop_rate = 0.1

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
    # model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(drop_rate))

    # L4 fully connected
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    model.summary()

    # model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=0)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

    return model


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

num_models = 5
models = []
losses = []
accuracies = []

for _ in range(num_models):
    model = fit_model(x_train, y_train)
    loss, acc = model.evaluate(x_test, y_test)

    models.append(model)
    losses.append(loss)
    accuracies.append(acc)

# make predictions
y_predicted = [model.predict(x_test) for model in models]
y_predicted = np.array(y_predicted)

# sum across ensembles
summed = np.sum(y_predicted, axis=0)

# argmax across classes
result = np.argmax(summed, axis=1)
print('result: ', result)

for _ in range(7):
    random_index = random.randint(0, x_test.shape[0] - 1)
    print('index: ', random_index,
          'actual y: ', np.argmax(y_test[random_index]),
          'predicted y: ', np.argmax(summed[random_index]))


# for model in models:
#     evaluation = model.evaluate(x_test, y_test)
#     losses.append(evaluation[0])
#     accuracies.append(evaluation[1])
#     print('loss: ', evaluation[0], ', accuracy: ', evaluation[1])

# ensemble_correct_prediction = tf.equal(result, np.argmax(y_test))
# ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32), 0)
# print('Ensemble accuracy: ', ensemble_accuracy)

print('mean_loss: ', np.mean(losses), ', mean_accuracy: ', np.mean(accuracies))
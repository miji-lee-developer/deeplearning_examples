import tensorflow as tf
import numpy as np
import pandas as pd

# y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=3)
data = pd.read_csv("./input/Titanic/train.csv", index_col="PassengerId")
print(data.shape)
data.head()

# print(data[["Embarked"]].head())
# embarked_one_hot = tf.keras.utils.to_categorical(data.Embarked, num_classes=3)
# print(embarked_one_hot)

t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_sequences(data.Embarked)
print(t.word_index)

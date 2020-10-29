import tensorflow as tf
import numpy as np
import pandas as pd

# y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=3)
data = pd.read_csv("./input/Titanic/train.csv", index_col="PassengerId")
print(data.shape)
data.head()

d = dict([e[:: -1] for e in enumerate(data.Embarked.unique())])
print(d)

data["Embarked_num"] = data.Embarked.map(d)
print(data[["Embarked", "Embarked_num"]])

one_hot = tf.keras.utils.to_categorical(data.Embarked_num, num_classes=len(d.items()))
print(one_hot)
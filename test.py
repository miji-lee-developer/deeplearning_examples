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

df = pd.DataFrame(one_hot)
data["Embarked_S"] = df[0]
data["Embarked_C"] = df[1]
data["Embarked_Q"] = df[2]
data["Embarked_Null"] = df[3]

print(df[0])
# print(data[["Embarked","Embarked_C","Embarked_S","Embarked_Q","Embarked_Null"]].head())
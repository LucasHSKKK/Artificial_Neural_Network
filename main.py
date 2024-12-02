import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv("dataset\Churn_Modelling.csv")
x = dataset.iloc[
    :, 3:-1
].values  # this takes since the third column till the one befor the ast one
y = dataset.iloc[:, -1].values  # this takes just the last one

# Encoding categorical data
# label encoding "gender"
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 2] = le.fit_transform(
    x[:, 2]
)  # this code turn the gender name column in numbers of 0 and 1

# one hot encoding "geography"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)  # turn the countries into number to facilitate the process
x = np.array(
    ct.fit_transform(x)
)  # The transformed dataset is converted into a NumPy array for easier manipulation

# Splitting the dataset into the traning set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)  # always fit the scaler on training set
x_test = sc.transform(x_test)  # use the same scaler to tranform the test set


# initializing the ANN
ann = tf.keras.models.Sequential()

# adding unput layer and first hidden layer
ann.add(
    tf.keras.layers.Dense(units=10, activation="relu")
)  # the units mean how many hidden layers (neruons) it will have, i can change if i  want

# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation="relu"))

# output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

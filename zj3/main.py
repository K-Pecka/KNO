import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime

df = pd.read_csv("../dataset/wine.csv")
df.reset_index()

data_names = ["Category", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
              "Flavanoids",
              "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
              "Proline"]

df.columns = data_names

df = df.sample(frac=1)

one_hot_encoded_data = pd.get_dummies(df, columns=["Category"], dtype="float")

X = one_hot_encoded_data[
    ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
     "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
     "Proline"]]

y = one_hot_encoded_data[["Category_1", "Category_2", "Category_3"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_1 = tf.keras.models.Sequential()
model_1.add(tf.keras.layers.Dense(units=64, activation='relu'))
model_1.add(tf.keras.layers.Dense(units=3, activation='softmax'))

model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                loss=tf.keras.losses.CategoricalCrossentropy, metrics=['accuracy'])

model_1.summary()

model_1_fit = model_1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

ModelLoss, ModelAccuracy = model_1.evaluate(X_test, y_test)

print(f'Test Loss is {ModelLoss}')
print(f'Test Accuracy is {ModelAccuracy}')

model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(156, activation='relu'),
    tf.keras.layers.Dropout(0.02),
    tf.keras.layers.Dense(156, activation='relu'),
    tf.keras.layers.Dropout(0.02),
    tf.keras.layers.Dense(156, activation='relu'),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(units=3, activation='softmax'),
])


model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_2.summary()
log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_2_fit = model_2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100,
                          callbacks=[tensorboard_callback],batch_size=32)

ModelLoss, ModelAccuracy = model_2.evaluate(X_test, y_test)

print(f'Test Loss is {ModelLoss}')
print(f'Test Accuracy is {ModelAccuracy}')

model_2.save('../model/zj3_model_2.keras')
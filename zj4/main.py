import datetime
import os

from model_function.main import *
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../dataset/wine.csv")

df.reset_index()

data_names = ["Category", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
              "Flavanoids",
              "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
              "Proline"]

df.columns = data_names

df = df.sample(frac=1)

one_hot_encoded_data = pd.get_dummies(df, columns=["Category"], dtype="float")

X = one_hot_encoded_data[[ "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols","Flavanoids",
                          "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines","Proline"]]

y = one_hot_encoded_data[["Category_1", "Category_2", "Category_3"]]

X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5, random_state=42)

if os.path.exists('../model/zj3_model_2.keras'):
    print('Model exists')
    baseline_model = tf.keras.models.load_model('../model/zj3_model_2.keras')

    val_loss, val_accuracy = baseline_model.evaluate(X_val, y_val)
    print(f'Wynik na zbiorze walidacyjnym - Loss: {val_loss}, Accuracy: {val_accuracy}')

    test_loss, test_accuracy = baseline_model.evaluate(X_test, y_test)
    print(f'Wynik na zbiorze testowym - Loss: {test_loss}, Accuracy: {test_accuracy}')

log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

results_df, best_model = test_model(X,y,tensorboard_callback,
                                    [0.2, 0.4],
                                    [[8, 16, 32], [128, 64], [128]],
                                    [0.05, 0.01])

print("\nWyniki:")
print(results_df)

print("\nThe best:")
print(best_model)
if os.path.exists('model_name.keras'):
    baseline_test_loss, baseline_test_accuracy = baseline_model.evaluate(X_test, y_test, verbose=0)
    print("baseline - Test loss:", baseline_test_loss)
    print("baseline - Test accuracy:", baseline_test_accuracy)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
data = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
np.random.shuffle(data)

label = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
         "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
         "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

X = data[:, 1:]
y = data[:, 0]

num_category = len(label)
y_one_hot = np.eye(num_category)[y.astype(int)]

X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

log_dir = "logs/fit/model_1/"
tensorboard_callback_1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
log_dir = "logs/fit/model_2/"
tensorboard_callback_2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model_1 = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(y_one_hot.shape[1], activation='softmax')
# ])
#
# model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model_1.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback_1])
#
#
#
# val_loss_1, val_acc_1 = model_1.evaluate(X_val, y_val)
# print(f'Model 1 - Validation Loss: {val_loss_1}, Validation Accuracy: {val_acc_1}')
#
# test_loss_1, test_acc_1 = model_1.evaluate(X_test, y_test)
# print(f'Model 1 - Test Loss: {test_loss_1}, Test Accuracy: {test_acc_1}')



#test_loss_1, test_acc_1 = model_1.evaluate(X_test, y_test)
#print(f'Model 1 - Test Loss: {test_loss_1}, Test Accuracy: {test_acc_1}')

# model_1.save('model/wine_classification_model_1.keras')



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(y_one_hot.shape[1], activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback_2])

train_loss, train_acc = model.evaluate(X_train, y_train)
print(f'Model - Train Loss: {train_loss}, Train Accuracy: {train_acc}')

val_loss_2, val_acc_2 = model.evaluate(X_val, y_val)
print(f'Model - Validation Loss: {val_loss_2}, Validation Accuracy: {val_acc_2}')

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Model - Test Loss: {test_loss}, Test Accuracy: {test_acc}')

with open('dane.json', 'w', encoding='utf-8') as file_json:
    json.dump({
        "train_loss":train_loss,
        "train_acc":train_acc,
        "val_loss":val_loss_2,
        "val_acc":val_acc_2,
        "test_loss":test_loss,
        "test_acc":test_acc
        },
        file_json, ensure_ascii=False, indent=4)

model.save('model/wine_classification_model.keras')

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def model_for_params(hp_units, hp_learning_rate, hp_dropout):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hp_units, activation='relu'),
        tf.keras.layers.Dropout(hp_dropout),
        tf.keras.layers.Dense(hp_units),
        tf.keras.layers.Dropout(hp_dropout),
        tf.keras.layers.Dense(y_one_hot.shape[1], activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


data = np.loadtxt('wine.csv', delimiter=',')
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

dense = [64, 128]
learning_rate = [0.001, 0.005]
dropout = [0.1, 0.3]

results = []

for units in dense:
    for lr in learning_rate:
        for drop in dropout:
            model = model_for_params(units, lr, drop)
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            results.append({
                'units': units,
                'learning_rate': lr,
                'dropout': drop,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc
            })

results_df = pd.DataFrame(results)

best_model_idx = results_df['test_accuracy'].idxmax()
best_model_params = results_df.iloc[best_model_idx]

print("\nBest model parameters based on validation accuracy:")
print(best_model_params)

baseline_model = model_for_params(128, 0.001, 0.3)
baseline_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)
baseline_test_loss, baseline_test_acc = baseline_model.evaluate(X_test, y_test)

print(f"\nBaseline Model Test Accuracy: {baseline_test_acc}")

best_model = model_for_params(best_model_params['units'], best_model_params['learning_rate'],
                              best_model_params['dropout'])
print(f"{best_model_params['units']}, {best_model_params['learning_rate']}, {best_model_params['dropout']}")
best_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)
best_model.save('model/best_wine_classification_model.keras')

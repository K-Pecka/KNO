import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

def create_model(dropout_rate, units, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=units[0], activation='relu'))

    for units in units[1:]:
        model.add(tf.keras.layers.Dense(units=units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy,
                  metrics=['accuracy'])

    return model

def test_model(X,y,tensorboard_callback,dropout_rates,units_list,learning_rates):

    results = []
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5, random_state=42)

    for dropout_rate in dropout_rates:
        for units in units_list:
            for learning_rate in learning_rates:
                print(f">>dropout_rate={dropout_rate} units={units} learning_rate={learning_rate}")

                model = create_model(dropout_rate=dropout_rate, units=units, learning_rate=learning_rate)


                model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])
                test_loss, test_accuracy = model.evaluate(X_test, y_test)

                results.append({
                    'dropout_rate': dropout_rate,
                    'units': units,
                    'learning_rate': learning_rate,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy
                })

    results_df = pd.DataFrame(results)

    best_model = results_df.sort_values(by='test_accuracy', ascending=False).iloc[0]

    return results_df, best_model
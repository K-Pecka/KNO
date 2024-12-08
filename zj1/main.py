import tensorflow as tf
import os
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist

def train():
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)
    model.save('model_name.keras')

def prediction(path,model_name):
    image = Image.open(path).convert('L')
    model = tf.keras.models.load_model(model_name)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    print(f"Model przewiduje: {predicted_class}")

if __name__ == '__main__':
    if os.path.exists('../../pythonProject2/model_name.keras'):
        prediction("1.png", 'model_name.keras')
    else:
        train()
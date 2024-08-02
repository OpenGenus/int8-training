import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        yield [tf.cast(input_value, tf.float32)]

if __name__ == "__main__":
    # Load dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create the model
    model = create_cnn_model()

    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')

    # Save the FP32 model
    model.save('cnn_fp32_model')

    # Convert the model to TensorFlow Lite INT8 with representative dataset
    converter = tf.lite.TFLiteConverter.from_saved_model('cnn_fp32_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_types = [tf.float32]
    converter.inference_input_type = tf.float32  # Keep input in FP32
    converter.inference_output_type = tf.float32  # Keep output in FP32

    tflite_model_int8 = converter.convert()

    # Save the quantized model
    with open('cnn_int8_model.tflite', 'wb') as f:
        f.write(tflite_model_int8)

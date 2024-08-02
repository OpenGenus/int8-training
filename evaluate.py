import numpy as np
import tensorflow as tf

# Load the quantized model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='cnn_int8_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def evaluate_model(interpreter, x_test, y_test):
    input_shape = input_details[0]['shape']
    correct_predictions = 0

    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)  # Keep input data as FP32
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output_data) == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(x_test)
    print(f'INT8 Model accuracy: {accuracy}')

if __name__ == "__main__":
    # Load the test dataset (assuming it was prepared in train_quantize.py)
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test[..., tf.newaxis]

    # Call the evaluation function with the test data
    evaluate_model(interpreter, x_test, y_test)

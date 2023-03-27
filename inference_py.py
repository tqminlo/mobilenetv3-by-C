import numpy as np
import tensorflow as tf
import cv2

IMG_PATH = "images/dog2.jpg"

# Load the TFLite model and allocate tensors.
MODEL_PATH = "weights/v3-large-minimalistic_224_1.0_uint8.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# print(input_shape)
input_data = cv2.imread(IMG_PATH)
input_data = cv2.resize(input_data, (224, 224))
input_data = np.expand_dims(input_data, 0)
print(input_data.shape)

input_type = input_details[0]['dtype']
print(input_type)
if input_type == np.uint8:
    input_scale, input_zero_point = input_details[0]['quantization']
    print("Input scale:", input_scale)
    print("Input zero point:", input_zero_point)
    # np_features = (np_features / input_scale) + input_zero_point
    # np_features = np.around(np_features)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
i = np.argmax(output_data[0])
print(i, output_data[0][i])
print(np.sum(output_data[0]))
print(len(output_data[0]))
for i in range(len(output_data[0])):
    print(output_data[0][i])


import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "weights/v3-large-minimalistic_224_1.0_uint8.tflite"
IMG_PATH = "images/lion1.jpg"
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = Image.open(IMG_PATH)
input_data = input_data.resize((224, 224))
input_data = np.array(input_data)
input_data = np.expand_dims(input_data, 0)
print(input_data.shape)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()


def check_original_predict():
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    i = np.argmax(output_data[0])
    print("id:", i, "score:", output_data[0][i])
    print(np.sum(output_data[0]))
    print(len(output_data[0]))


def check_same_out():
    '''
    Check output of original model and merge model

    Original:
        Tensor before pw29:
            name    : MobilenetV3/expanded_conv_14/add
            shape   : (1, 7, 7, 160)
            index   : 71

        Tensor after pw29 (before argpool0):
            name    : MobilenetV3/Conv_1/Relu
            shape   : (1, 7, 7, 960)
            index   : 5
    Merge:

    '''

    tensor = interpreter.get_tensor(5)
    print(tensor.shape)
    print(np.max(tensor))
    print(np.min(tensor))
    num = 0
    for i in range(len(tensor[0])):
        for j in range(len(tensor[0][i])):
            for k in range(len(tensor[0][i][j])):
                if tensor[0][i][j][k] == 0:
                    num += 1
    print(num)

    tensor_float = tensor * 0.4507608115673065
    print(np.max(tensor_float))
    print(np.min(tensor_float))


def check_pw29():
    pw29_w_path = "weights/npy/tensor (90).npy"
    pw29_b_path = "weights/npy/tensor (91).npy"
    pw29_w_Z = 134
    pw29_w_S = 0.003714228980243206
    pw29_b_S = 0.0018642294453456998

    pw29_w = np.load(pw29_w_path).astype(np.float64)
    pw29_b = np.load(pw29_b_path).astype(np.float64)
    pw29_w = pw29_w.reshape(960, 160)
    print(pw29_w.shape)
    print(pw29_b.shape)

    tensor = interpreter.get_tensor(71)
    # tensor = tensor.astype(np.float64)
    print(tensor.shape)
    tensor_Z = 126
    tensor_S = 0.5019155740737915
    # print(np.max(tensor))
    # print(np.min(tensor))

    sum = 0
    out = np.zeros(shape=(7, 7, 960), dtype=np.float64)
    for i in range(7):
        for j in range(7):
            out[i][j] = ((pw29_w - pw29_w_Z) * pw29_w_S).dot((tensor[0][i][j] - tensor_Z) * tensor_S) + pw29_b * pw29_b_S
            for k in range(len(out[i][j])):
                if out[i][j][k] < 0:
                    # print(i, j, k, out[i][j][k])
                    sum += 1
    print(sum)
    print(np.max(out))
    print(np.min(out))


def check_pw30():
    pw30_w_path = "weights/npy/tensor (92).npy"
    pw30_b_path = "weights/npy/tensor (93).npy"
    pw30_w_Z = 93
    pw30_w_S = 0.008932989090681076
    pw30_b_S = 0.004026641603559256

    pw30_w = np.load(pw30_w_path).astype(np.int64)
    pw30_b = np.load(pw30_b_path).astype(np.int64)
    pw30_w = pw30_w.reshape(1280, 960)
    print(pw30_w.shape)
    print(pw30_b.shape)

    inp = interpreter.get_tensor(0)
    print('inp max:', np.max(inp), 'inp min:', np.min(inp))
    print("inp shape:", inp.shape)
    inp.astype(np.int64)
    inp_Z = 0
    inp_S = 0.019861046224832535

    sum = 0
    # out = np.zeros(shape=(1280,), dtype=np.float64)
    out = ((pw30_w - pw30_w_Z) * pw30_w_S).dot((inp[0][0][0] - inp_Z) * inp_S) + pw30_b * pw30_b_S
    for k in range(len(out)):
        if out[k] < 0:
            # print(k, out[k])
            sum += 1
    print(sum)
    print(np.max(out))
    print(np.min(out))

    out = interpreter.get_tensor(8)
    print('out max:', np.max(out), 'out min:', np.min(out))
    print("out shape:", out.shape)

    check = interpreter.get_tensor(5)
    print('check max:', np.max(check), 'check min:', np.min(check))
    print("check shape:", check.shape)

    check2 = interpreter.get_tensor(11)
    print('check2 max:', np.max(check2), 'check2 min:', np.min(check2))
    print("check2 shape:", check2.shape)

if __name__ == "__main__":
    # check_original_predict()
    # check_same_out()
    # check_pw29()
    check_pw30()
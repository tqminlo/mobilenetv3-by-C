import os
import numpy as np


def npy_to_txt():
    npy_dir = "weights/npy"
    txt_dir = "weights/txt"
    os.makedirs(txt_dir, exist_ok=True)

    npys = os.listdir(npy_dir)
    id_cv = -1
    id_dw = -1
    id_pw = -1
    for i in range(0, len(npys), 2):
        '''i follow weight tensor'''
        npy = f"tensor ({i}).npy"
        txt = ""
        layer = ""
        npy_path = os.path.join(npy_dir, npy)
        tensor = np.load(npy_path)
        shape = tensor.shape
        assert len(shape) == 4, "false at anything"
        if shape[0] == 1:
            id_dw += 1
            layer = f"dw{id_dw}"
        elif shape[1] == 1:
            id_pw += 1
            layer = f"pw{id_pw}"
        else:
            id_cv += 1
            layer = f"cv{id_cv}"

        '''process weight tensor'''
        txt = f"{layer}_weight_{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}.txt"
        txt_path = os.path.join(txt_dir, txt)
        tensor = tensor.flatten()
        tensor = list(tensor)
        tensor_text = str(tensor)
        tensor_text = "{" + tensor_text[1:-1] + "}"
        with open(txt_path, "w") as f:
            f.write(tensor_text)

        '''process bias tensor, just follow weight tensor'''
        npy_bias = f"tensor ({i+1}).npy"
        npy_bias_path = os.path.join(npy_dir, npy_bias)
        tensor_bias = np.load(npy_bias_path)
        shape_bias = tensor_bias.shape
        assert len(shape_bias) == 1, "false at anything bias"
        txt_bias = f"{layer}_bias_{shape_bias[0]}.txt"
        txt_bias_path = os.path.join(txt_dir, txt_bias)
        tensor_bias = list(tensor_bias)
        tensor_bias_text = str(tensor_bias)
        tensor_bias_text = "{" + tensor_bias_text[1:-1] + "}"
        with open(txt_bias_path, "w") as f:
            f.write(tensor_bias_text)


def txt_to_weight_code():
    txt_dir = "weights/txt"
    txtcode = "weights.c"
    txts = os.listdir(txt_dir)

    with open(txtcode, "w") as f:
        for txt in txts:
            txt_path = os.path.join(txt_dir, txt)
            with open(txt_path, "r") as f1:
                head = "int " + txt.split('_')[1] + "_" + txt.split('_')[0] + f"[{txt.split('_')[2].split('.')[0].replace('x', '*')}] = "
                content = f1.readline()
                f1.close()
            line = head + content + ";\n\n"
            f.write(line)


def txt_to_main_code():
    txt_dir = "weights/txt"
    main_code = "inference_v2.c"
    txts = os.listdir(txt_dir)

    with open(main_code, "w") as f:
        for txt in txts:
            line = "int extern " + txt.split('_')[1] + "_" + txt.split('_')[0] + f"[{txt.split('_')[2].split('.')[0].replace('x', '*')}];\n"
            f.write(line)



if __name__ == "__main__":
    npy_to_txt()
    txt_to_weight_code()
    txt_to_main_code()

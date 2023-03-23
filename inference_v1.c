#include <stdio.h>
#include <stdlib.h>
#include <math.h>



// Define input
float input[224][224][3];
void init_input() {
    for (int i = 0; i < 224; i++) {
        for (int j = 0; j < 224; j++) {
            for (int k = 0; k < 3; k++) {
                input[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }
}


// Define ouput-conv0
float output_conv0[112][112][16];
// Conv2d-0, padding, bias=False, stride=2, relu
void convolution2d_0(){
    // padding input
    static float input_pad[225][225][3] = {0};
    for (int i = 0; i < 224; i++) {
        for (int j = 0; j < 224; j++) {
            for (int k = 0; k < 3; k++) {
                input_pad[i][j][k] = input[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][3][16];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 16; l++) {
                    kernel[i][j][k][l] = (float) rand() / (float) RAND_MAX;
                }
            }
        }
    }

    // Define bias tensor
    float bias[16];
    // Load bias
    for (int i = 0; i < 16; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Conv2d
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int l = 0; l < 16; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        for (int k = 0; k < 3; k++) {
                            sum += input_pad[i*2 + ii][j*2 + jj][k] * kernel[ii][jj][k][l];  // stride = 2
                        }
                    }
                }
                output_conv0[i][j][l] = sum + bias[l];  // bias
                if (output_conv0[i][j][l]<0) output_conv0[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw0
float output_dw0[112][112][16];
void depthwise_0(){
    // padding input this layer
    static float input_pad[114][114][16] = {0};
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int k = 0; k < 16; k++) {
                input_pad[i + 1][j + 1][k] = output_conv0[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][16];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 16; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[16];
    // Load bias
    for (int i = 0; i < 16; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int l = 0; l < 16; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 2
                    }
                }
                output_dw0[i][j][l] = sum + bias[l];  // bias
                if (output_dw0[i][j][l]<0) output_dw0[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw0
float output_pw0[112][112][16];
// Piecewise, no bias
void piecewise_0(){
    // Define kernel tensor
    float kernel[16][16];
    // Load kernel
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[16];
    // Load bias
    for (int i = 0; i < 16; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int l = 0; l < 16; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 16; k++){
                    sum += output_dw0[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw0[i][j][l] = sum + bias[l];  // no bias
                if (output_pw0[i][j][l]<0) output_pw0[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-add0, connect output_conv0 vs output_pw0
float output_add0[112][112][16];
void add_0(){
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int l = 0; l < 16; l++) {
                output_add0[i][j][l] = output_conv0[i][j][l] + output_pw0[i][j][l];
            }
        }
    }
}


// Define ouput-pw1
float output_pw1[112][112][64];
// Piecewise, bias
void piecewise_1(){
    // Define kernel tensor
    float kernel[64][16];
    // Load kernel
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 16; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[64];
    // Load bias
    for (int i = 0; i < 64; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int l = 0; l < 64; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 16; k++){
                    sum += output_add0[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw1[i][j][l] = sum + bias[l];  // bias
                if (output_pw1[i][j][l]<0) output_pw1[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw1
float output_dw1[56][56][64];
void depthwise_1(){
    // padding input this layer
    static float input_pad[113][113][64] = {0};
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int k = 0; k < 64; k++) {
                input_pad[i][j][k] = output_pw1[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][64];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 64; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[64];
    // Load bias
    for (int i = 0; i < 64; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int l = 0; l < 64; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i*2 + ii][j*2 + jj][l] * kernel[ii][jj][l];  // stride = 2
                    }
                }
                output_dw1[i][j][l] = sum + bias[l];  // bias
                if (output_dw1[i][j][l]<0) output_dw1[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw2
float output_pw2[56][56][24];
// Piecewise, bias, no relu
void piecewise_2(){
    // Define kernel tensor
    float kernel[24][64];
    // Load kernel
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 64; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[24];
    // Load bias
    for (int i = 0; i < 24; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int l = 0; l < 24; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 64; k++){
                    sum += output_dw1[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw2[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-pw3
float output_pw3[56][56][72];
// Piecewise, bias
void piecewise_3(){
    // Define kernel tensor
    float kernel[72][24];
    // Load kernel
    for (int i = 0; i < 72; i++) {
        for (int j = 0; j < 24; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[72];
    // Load bias
    for (int i = 0; i < 72; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int l = 0; l < 72; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 24; k++){
                    sum += output_pw2[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw3[i][j][l] = sum + bias[l];  // no bias
                if (output_pw3[i][j][l]<0) output_pw3[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw2
float output_dw2[56][56][72];
void depthwise_2(){
    // padding input this layer
    static float input_pad[58][58][72] = {0};
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int k = 0; k < 72; k++) {
                input_pad[i][j][k] = output_pw3[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][72];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 72; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[72];
    // Load bias
    for (int i = 0; i < 72; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int l = 0; l < 72; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw2[i][j][l] = sum + bias[l];  // bias
                if (output_dw2[i][j][l]<0) output_dw2[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw4
float output_pw4[56][56][24];
// Piecewise, bias, no relu
void piecewise_4(){
    // Define kernel tensor
    float kernel[24][72];
    // Load kernel
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 72; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[24];
    // Load bias
    for (int i = 0; i < 24; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int l = 0; l < 24; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 72; k++) {
                    sum += output_dw2[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw4[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add1, connect output_pw2 vs output_pw4
float output_add1[56][56][24];
void add_1(){
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int l = 0; l < 24; l++) {
                output_add1[i][j][l] = output_pw2[i][j][l] + output_pw4[i][j][l];
            }
        }
    }
}


// Define ouput-pw5
float output_pw5[56][56][72];
// Piecewise, bias
void piecewise_5(){
    // Define kernel tensor
    float kernel[72][24];
    // Load kernel
    for (int i = 0; i < 72; i++) {
        for (int j = 0; j < 24; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[72];
    // Load bias
    for (int i = 0; i < 72; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int l = 0; l < 72; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 24; k++){
                    sum += output_add1[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw5[i][j][l] = sum + bias[l];  // no bias
                if (output_pw5[i][j][l]<0) output_pw5[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw3
float output_dw3[28][28][72];
void depthwise_3(){
    // padding input this layer
    static float input_pad[57][57][72] = {0};
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int k = 0; k < 72; k++) {
                input_pad[i][j][k] = output_pw3[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][72];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 72; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[72];
    // Load bias
    for (int i = 0; i < 72; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 72; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i*2 + ii][j*2 + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw3[i][j][l] = sum + bias[l];  // bias
                if (output_dw3[i][j][l]<0) output_dw3[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw6
float output_pw6[28][28][40];
// Piecewise, bias, no relu
void piecewise_6(){
    // Define kernel tensor
    float kernel[40][72];
    // Load kernel
    for (int i = 0; i < 40; i++) {
        for (int j = 0; j < 72; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[40];
    // Load bias
    for (int i = 0; i < 40; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 40; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 72; k++) {
                    sum += output_dw3[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw6[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-pw7
float output_pw7[28][28][120];
// Piecewise, bias
void piecewise_7(){
    // Define kernel tensor
    float kernel[120][40];
    // Load kernel
    for (int i = 0; i < 120; i++) {
        for (int j = 0; j < 40; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[120];
    // Load bias
    for (int i = 0; i < 120; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 120; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 40; k++){
                    sum += output_pw6[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw7[i][j][l] = sum + bias[l];  // bias
                if (output_pw7[i][j][l]<0) output_pw7[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw4
float output_dw4[28][28][120];
void depthwise_4(){
    // padding input this layer
    static float input_pad[30][30][120] = {0};
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 120; k++) {
                input_pad[i][j][k] = output_pw7[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][120];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 120; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[120];
    // Load bias
    for (int i = 0; i < 120; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 120; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw4[i][j][l] = sum + bias[l];  // bias
                if (output_dw4[i][j][l]<0) output_dw4[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw8
float output_pw8[28][28][40];
// Piecewise, bias, no relu
void piecewise_8(){
    // Define kernel tensor
    float kernel[40][120];
    // Load kernel
    for (int i = 0; i < 40; i++) {
        for (int j = 0; j < 120; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[40];
    // Load bias
    for (int i = 0; i < 40; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 40; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 120; k++) {
                    sum += output_dw4[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw8[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add2, connect output_pw6 vs output_pw8
float output_add2[28][28][40];
void add_2(){
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 40; l++) {
                output_add2[i][j][l] = output_pw6[i][j][l] + output_pw8[i][j][l];
            }
        }
    }
}


// Define ouput-pw9
float output_pw9[28][28][120];
// Piecewise, bias
void piecewise_9(){
    // Define kernel tensor
    float kernel[120][40];
    // Load kernel
    for (int i = 0; i < 120; i++) {
        for (int j = 0; j < 40; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[120];
    // Load bias
    for (int i = 0; i < 120; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 120; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 40; k++){
                    sum += output_add2[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw9[i][j][l] = sum + bias[l];  // bias
                if (output_pw9[i][j][l]<0) output_pw9[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw5
float output_dw5[28][28][120];
void depthwise_5(){
    // padding input this layer
    static float input_pad[30][30][120] = {0};
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 120; k++) {
                input_pad[i][j][k] = output_pw9[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][120];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 120; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[120];
    // Load bias
    for (int i = 0; i < 120; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 120; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw5[i][j][l] = sum + bias[l];  // bias
                if (output_dw5[i][j][l]<0) output_dw5[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw10
float output_pw10[28][28][40];
// Piecewise, bias, no relu
void piecewise_10(){
    // Define kernel tensor
    float kernel[40][120];
    // Load kernel
    for (int i = 0; i < 40; i++) {
        for (int j = 0; j < 120; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[40];
    // Load bias
    for (int i = 0; i < 40; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 40; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 120; k++) {
                    sum += output_dw5[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw10[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add3, connect output_add2 vs output_pw10
float output_add3[28][28][40];
void add_3(){
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 40; l++) {
                output_add3[i][j][l] = output_add2[i][j][l] + output_pw10[i][j][l];
            }
        }
    }
}


// Scale 1
void scale0(){
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 40; l++) {
                output_add3[i][j][l] = output_add3[i][j][l] / 10000000;
            }
        }
    }
}


// Define ouput-pw11
float output_pw11[28][28][240];
// Piecewise, bias
void piecewise_11(){
    // Define kernel tensor
    float kernel[240][40];
    // Load kernel
    for (int i = 0; i < 240; i++) {
        for (int j = 0; j < 40; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[240];
    // Load bias
    for (int i = 0; i < 120; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int l = 0; l < 240; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 40; k++){
                    sum += output_add3[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw11[i][j][l] = sum + bias[l];  // bias
                if (output_pw11[i][j][l]<0) output_pw11[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw6
float output_dw6[14][14][240];
void depthwise_6(){
    // padding input this layer
    static float input_pad[29][29][240] = {0};
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 240; k++) {
                input_pad[i][j][k] = output_pw11[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][240];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 240; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[240];
    // Load bias
    for (int i = 0; i < 240; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 240; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i*2 + ii][j*2 + jj][l] * kernel[ii][jj][l];  // stride = 2
                    }
                }
                output_dw6[i][j][l] = sum + bias[l];  // bias
                if (output_dw6[i][j][l]<0) output_dw6[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw12
float output_pw12[14][14][80];
// Piecewise, bias, no relu
void piecewise_12(){
    // Define kernel tensor
    float kernel[80][240];
    // Load kernel
    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < 240; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[80];
    // Load bias
    for (int i = 0; i < 80; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 240; k++) {
                    sum += output_dw6[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw12[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-pw13
float output_pw13[14][14][200];
// Piecewise, bias
void piecewise_13(){
    // Define kernel tensor
    float kernel[200][80];
    // Load kernel
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 80; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[200];
    // Load bias
    for (int i = 0; i < 200; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 200; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 80; k++){
                    sum += output_pw12[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw13[i][j][l] = sum + bias[l];  // bias
                if (output_pw13[i][j][l]<0) output_pw13[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw7
float output_dw7[14][14][200];
void depthwise_7(){
    // padding input this layer
    static float input_pad[16][16][200] = {0};
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int k = 0; k < 200; k++) {
                input_pad[i][j][k] = output_pw13[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][200];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 200; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[200];
    // Load bias
    for (int i = 0; i < 200; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 200; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw7[i][j][l] = sum + bias[l];  // bias
                if (output_dw7[i][j][l]<0) output_dw7[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw14
float output_pw14[14][14][80];
// Piecewise, bias, no relu
void piecewise_14(){
    // Define kernel tensor
    float kernel[80][240];
    // Load kernel
    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < 240; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[80];
    // Load bias
    for (int i = 0; i < 80; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 240; k++) {
                    sum += output_dw7[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw14[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add4, connect output_pw12 vs output_pw14
float output_add4[14][14][80];
void add_4(){
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                output_add4[i][j][l] = output_pw12[i][j][l] + output_pw14[i][j][l];
            }
        }
    }
}


// Define ouput-pw15
float output_pw15[14][14][184];
// Piecewise, bias
void piecewise_15(){
    // Define kernel tensor
    float kernel[184][80];
    // Load kernel
    for (int i = 0; i < 184; i++) {
        for (int j = 0; j < 80; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[184];
    // Load bias
    for (int i = 0; i < 184; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 184; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 80; k++){
                    sum += output_add4[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw15[i][j][l] = sum + bias[l];  // bias
                if (output_pw15[i][j][l]<0) output_pw15[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw8
float output_dw8[14][14][184];
void depthwise_8(){
    // padding input this layer
    static float input_pad[16][16][184] = {0};
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int k = 0; k < 184; k++) {
                input_pad[i][j][k] = output_pw15[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][184];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 184; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[184];
    // Load bias
    for (int i = 0; i < 184; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 184; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw8[i][j][l] = sum + bias[l];  // bias
                if (output_dw8[i][j][l]<0) output_dw8[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw16
float output_pw16[14][14][80];
// Piecewise, bias, no relu
void piecewise_16(){
    // Define kernel tensor
    float kernel[80][184];
    // Load kernel
    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < 184; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[80];
    // Load bias
    for (int i = 0; i < 80; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 184; k++) {
                    sum += output_dw8[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw16[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add5, connect output_add4 vs output_pw16
float output_add5[14][14][80];
void add_5(){
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                output_add5[i][j][l] = output_add4[i][j][l] + output_pw16[i][j][l];
            }
        }
    }
}


// Define ouput-pw17
float output_pw17[14][14][184];
// Piecewise, bias
void piecewise_17(){
    // Define kernel tensor
    float kernel[184][80];
    // Load kernel
    for (int i = 0; i < 184; i++) {
        for (int j = 0; j < 80; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[184];
    // Load bias
    for (int i = 0; i < 184; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 184; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 80; k++){
                    sum += output_add5[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw17[i][j][l] = sum + bias[l];  // bias
                if (output_pw17[i][j][l]<0) output_pw17[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw9
float output_dw9[14][14][184];
void depthwise_9(){
    // padding input this layer
    static float input_pad[16][16][184] = {0};
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int k = 0; k < 184; k++) {
                input_pad[i][j][k] = output_pw17[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][184];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 184; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[184];
    // Load bias
    for (int i = 0; i < 184; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 184; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw9[i][j][l] = sum + bias[l];  // bias
                if (output_dw9[i][j][l]<0) output_dw9[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw18
float output_pw18[14][14][80];
// Piecewise, bias, no relu
void piecewise_18(){
    // Define kernel tensor
    float kernel[80][184];
    // Load kernel
    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < 184; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[80];
    // Load bias
    for (int i = 0; i < 80; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 184; k++) {
                    sum += output_dw9[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw18[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add6, connect output_add5 vs output_pw18
float output_add6[14][14][80];
void add_6(){
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                output_add6[i][j][l] = output_add5[i][j][l] + output_pw18[i][j][l];
            }
        }
    }
}


void scale1(){
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 80; l++) {
                output_add6[i][j][l] = output_add6[i][j][l] / 10000000;
            }
        }
    }
}


// Define ouput-pw19
float output_pw19[14][14][480];
// Piecewise, bias
void piecewise_19(){
    // Define kernel tensor
    float kernel[480][80];
    // Load kernel
    for (int i = 0; i < 480; i++) {
        for (int j = 0; j < 80; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[480];
    // Load bias
    for (int i = 0; i < 480; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 480; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 80; k++){
                    sum += output_add6[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw19[i][j][l] = sum + bias[l];  // bias
                if (output_pw19[i][j][l]<0) output_pw19[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw10
float output_dw10[14][14][480];
void depthwise_10(){
    // padding input this layer
    static float input_pad[16][16][480] = {0};
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int k = 0; k < 480; k++) {
                input_pad[i][j][k] = output_pw19[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][480];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 480; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[480];
    // Load bias
    for (int i = 0; i < 480; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 480; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw10[i][j][l] = sum + bias[l];  // bias
                if (output_dw10[i][j][l]<0) output_dw10[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw20
float output_pw20[14][14][112];
// Piecewise, bias, no relu
void piecewise_20(){
    // Define kernel tensor
    float kernel[112][480];
    // Load kernel
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 480; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[112];
    // Load bias
    for (int i = 0; i < 112; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 112; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 480; k++) {
                    sum += output_dw10[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw20[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-pw21
float output_pw21[14][14][672];
// Piecewise, bias
void piecewise_21(){
    // Define kernel tensor
    float kernel[672][112];
    // Load kernel
    for (int i = 0; i < 672; i++) {
        for (int j = 0; j < 112; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[672];
    // Load bias
    for (int i = 0; i < 672; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 672; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 112; k++){
                    sum += output_pw20[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw21[i][j][l] = sum + bias[l];  // bias
                if (output_pw21[i][j][l]<0) output_pw21[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw11
float output_dw11[14][14][672];
void depthwise_11(){
    // padding input this layer
    static float input_pad[16][16][672] = {0};
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int k = 0; k < 672; k++) {
                input_pad[i][j][k] = output_pw21[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][672];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 672; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[672];
    // Load bias
    for (int i = 0; i < 672; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 672; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw11[i][j][l] = sum + bias[l];  // bias
                if (output_dw11[i][j][l]<0) output_dw11[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw22
float output_pw22[14][14][112];
// Piecewise, bias, no relu
void piecewise_22(){
    // Define kernel tensor
    float kernel[112][672];
    // Load kernel
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 672; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[112];
    // Load bias
    for (int i = 0; i < 112; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 112; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 672; k++) {
                    sum += output_dw11[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw22[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add7, connect output_pw20 vs output_pw22
float output_add7[14][14][112];
void add_7(){
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 112; l++) {
                output_add7[i][j][l] = output_pw20[i][j][l] + output_pw22[i][j][l];
            }
        }
    }
}


// Define ouput-pw23
float output_pw23[14][14][672];
// Piecewise, bias
void piecewise_23(){
    // Define kernel tensor
    float kernel[672][112];
    // Load kernel
    for (int i = 0; i < 672; i++) {
        for (int j = 0; j < 112; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[672];
    // Load bias
    for (int i = 0; i < 672; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int l = 0; l < 672; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 112; k++){
                    sum += output_add7[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw23[i][j][l] = sum + bias[l];  // bias
                if (output_pw23[i][j][l]<0) output_pw23[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw12
float output_dw12[7][7][672];
void depthwise_12(){
    // padding input this layer
    static float input_pad[15][15][672] = {0};
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int k = 0; k < 672; k++) {
                input_pad[i][j][k] = output_pw23[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][672];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 672; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[672];
    // Load bias
    for (int i = 0; i < 672; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 672; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i*2 + ii][j*2 + jj][l] * kernel[ii][jj][l];  // stride = 2
                    }
                }
                output_dw12[i][j][l] = sum + bias[l];  // bias
                if (output_dw12[i][j][l]<0) output_dw12[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw24
float output_pw24[7][7][160];
// Piecewise, bias, no relu
void piecewise_24(){
    // Define kernel tensor
    float kernel[160][672];
    // Load kernel
    for (int i = 0; i < 160; i++) {
        for (int j = 0; j < 672; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[160];
    // Load bias
    for (int i = 0; i < 160; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 160; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 672; k++) {
                    sum += output_dw12[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw24[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-pw25
float output_pw25[7][7][960];
// Piecewise, bias
void piecewise_25(){
    // Define kernel tensor
    float kernel[960][160];
    // Load kernel
    for (int i = 0; i < 960; i++) {
        for (int j = 0; j < 160; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[960];
    // Load bias
    for (int i = 0; i < 960; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 960; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 160; k++){
                    sum += output_pw24[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw25[i][j][l] = sum + bias[l];  // bias
                if (output_pw25[i][j][l]<0) output_pw25[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw13
float output_dw13[7][7][960];
void depthwise_13(){
    // padding input this layer
    static float input_pad[9][9][960] = {0};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            for (int k = 0; k < 960; k++) {
                input_pad[i+1][j+1][k] = output_pw25[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][960];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 960; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[960];
    // Load bias
    for (int i = 0; i < 960; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 960; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw13[i][j][l] = sum + bias[l];  // bias
                if (output_dw13[i][j][l]<0) output_dw13[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw26
float output_pw26[7][7][160];
// Piecewise, bias, no relu
void piecewise_26(){
    // Define kernel tensor
    float kernel[160][960];
    // Load kernel
    for (int i = 0; i < 160; i++) {
        for (int j = 0; j < 960; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[160];
    // Load bias
    for (int i = 0; i < 160; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 160; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 960; k++) {
                    sum += output_dw13[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw26[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add8, connect output_pw24 vs output_pw26
float output_add8[7][7][160];
void add_8(){
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 160; l++) {
                output_add8[i][j][l] = output_pw24[i][j][l] + output_pw26[i][j][l];
            }
        }
    }
}


// Define ouput-pw27
float output_pw27[7][7][960];
// Piecewise, bias
void piecewise_27(){
    // Define kernel tensor
    float kernel[960][160];
    // Load kernel
    for (int i = 0; i < 960; i++) {
        for (int j = 0; j < 160; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[960];
    // Load bias
    for (int i = 0; i < 960; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 960; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 160; k++){
                    sum += output_pw26[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw27[i][j][l] = sum + bias[l];  // bias
                if (output_pw27[i][j][l]<0) output_pw27[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-dw14
float output_dw14[7][7][960];
void depthwise_14(){
    // padding input this layer
    static float input_pad[9][9][960] = {0};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            for (int k = 0; k < 960; k++) {
                input_pad[i+1][j+1][k] = output_pw27[i][j][k];
            }
        }
    }

    // Define kernel tensor
    float kernel[3][3][960];
    // Load kernel
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 960; k++) {
                kernel[i][j][k] = (float) rand() / (float) RAND_MAX;
            }
        }
    }

    // Define bias tensor
    float bias[960];
    // Load bias
    for (int i = 0; i < 960; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }
//
    // Dw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 960; l++) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        sum += input_pad[i + ii][j + jj][l] * kernel[ii][jj][l];  // stride = 1
                    }
                }
                output_dw14[i][j][l] = sum + bias[l];  // bias
                if (output_dw14[i][j][l]<0) output_dw14[i][j][l] = 0;  // relu
            }
        }
    }
}


// Define ouput-pw28
float output_pw28[7][7][160];
// Piecewise, bias, no relu
void piecewise_28(){
    // Define kernel tensor
    float kernel[160][960];
    // Load kernel
    for (int i = 0; i < 160; i++) {
        for (int j = 0; j < 960; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[160];
    // Load bias
    for (int i = 0; i < 160; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 160; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 960; k++) {
                    sum += output_dw14[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw28[i][j][l] = sum + bias[l];  // bias
            }
        }
    }
}


// Define ouput-add8, connect output_add8 vs output_pw28
float output_add9[7][7][160];
void add_9(){
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 160; l++) {
                output_add9[i][j][l] = output_add8[i][j][l] + output_pw28[i][j][l];
            }
        }
    }
}


void scale2(){
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 160; l++) {
                output_add9[i][j][l] = output_add9[i][j][l] / 10000000;
            }
        }
    }
}


// Define ouput-pw29
float output_pw29[7][7][960];
// Piecewise, bias
void piecewise_29(){
    // Define kernel tensor
    float kernel[960][160];
    // Load kernel
    for (int i = 0; i < 960; i++) {
        for (int j = 0; j < 160; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[960];
    // Load bias
    for (int i = 0; i < 960; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Pw
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 960; l++) {
                float sum = 0.0f;
                for (int k = 0; k < 160; k++){
                    sum += output_add9[i][j][k] * kernel[l][k];  // stride = 1
                }
                output_pw29[i][j][l] = sum + bias[l];  // bias
                if (output_pw29[i][j][l]<0) output_pw29[i][j][l] = 0;  // relu
            }
        }
    }
}


float output_avgpool0[960];
void averagepool_0(){
    for (int i = 0; i < 960; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 7; j++) {
            for (int l = 0; l < 7; l++){
                sum += output_pw29[j][l][i];
            }
        }
        output_avgpool0[i] = sum / 49;
    }
}


// Define output_dense0
float output_dense0[1280];
void dense_0(){
    // Define kernel tensor
    static float kernel[1280][960];
    // Load kernel
    for (int i = 0; i < 1280; i++) {
        for (int j = 0; j < 960; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[1280];
    // Load bias
    for (int i = 0; i < 1280; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dense
    for (int i = 0; i < 1280; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 960; j++){
            sum += output_avgpool0[j] * kernel[i][j];  // stride = 1
        }
        output_dense0[i] = sum + bias[i];  // bias
    }
}


// Define output_dense1
float output_dense1[1001];
void dense_1(){
    // Define kernel tensor
    static float kernel[1001][1280];
    // Load kernel
    for (int i = 0; i < 1001; i++) {
        for (int j = 0; j < 1280; j++) {
            kernel[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }

    // Define bias tensor
    float bias[1001];
    // Load bias
    for (int i = 0; i < 1001; i++) {
        bias[i] = (float) rand() / (float) RAND_MAX;
    }

    // Dense
    for (int i = 0; i < 1001; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 1280; j++){
            sum += output_dense0[j] * kernel[i][j];  // stride = 1
        }
        output_dense1[i] = sum + bias[i];  // bias
    }
}


void scale3(){
    for (int i = 0; i < 1001; i++) {
        output_dense1[i] = output_dense1[i] / 100000;
    }
}


// Define output_softmax0
float output_softmax0[1001];
void softmax_0(){
    // Softmax
    float mid_array[1001];
    float sum = 0.0f;
    for (int i = 0; i < 1001; i++) {
        mid_array[i] = exp(output_dense1[i]);
        sum += mid_array[i];
    }
    for (int i = 0; i < 1001; i++) output_softmax0[i] = mid_array[i] / sum;
}



int main() {
    init_input();
    printf("%f\n", input[2][2][2]);
    convolution2d_0();
    printf("%f\n", output_conv0[2][2][2]);
    depthwise_0();
    printf("%f\n", output_dw0[2][2][2]);
    piecewise_0();
    printf("%f\n", output_pw0[2][2][2]);
    add_0();
    printf("%f\n", output_add0[2][2][2]);
    piecewise_1();
    printf("%f\n", output_pw1[2][2][2]);
    depthwise_1();
    printf("%f\n", output_dw1[2][2][2]);
    piecewise_2();
    printf("%f\n", output_pw2[2][2][2]);
    piecewise_3();
    printf("%f\n", output_pw3[2][2][2]);
    depthwise_2();
    printf("%f\n", output_dw2[2][2][2]);
    piecewise_4();
    printf("%f\n", output_pw4[2][2][2]);
    add_1();
    printf("%f\n", output_add1[2][2][2]);
    piecewise_5();
    printf("%f\n", output_pw5[2][2][2]);
    depthwise_3();
    printf("%f\n", output_dw3[2][2][2]);
    piecewise_6();
    printf("%f\n", output_pw6[2][2][2]);
    piecewise_7();
    printf("%f\n", output_pw7[2][2][2]);
    depthwise_4();
    printf("%f\n", output_dw4[2][2][2]);
    piecewise_8();
    printf("%f\n", output_pw8[2][2][2]);
    add_2();
    printf("%f\n", output_add2[2][2][2]);
    piecewise_9();
    printf("%f\n", output_pw9[2][2][2]);
    depthwise_5();
    printf("%f\n", output_dw5[2][2][2]);
    piecewise_10();
    printf("%f\n", output_pw10[2][2][2]);
    add_3();
    printf("%f  ", output_add3[2][2][2]);
    scale0();
    scale0();
    printf("%f\n", output_add3[2][2][2]);
    piecewise_11();
    printf("%f\n", output_pw11[2][2][2]);
    depthwise_6();
    printf("%f\n", output_dw6[2][2][2]);
    piecewise_12();
    printf("%f\n", output_pw12[2][2][2]);
    piecewise_13();
    printf("%f\n", output_pw13[2][2][2]);
    depthwise_7();
    printf("%f\n", output_dw7[2][2][2]);
    piecewise_14();
    printf("%f\n", output_pw14[2][2][2]);
    add_4();
    printf("%f\n", output_add4[2][2][2]);
    piecewise_15();
    printf("%f\n", output_pw15[2][2][2]);
    depthwise_8();
    printf("%f\n", output_dw8[2][2][2]);
    piecewise_16();
    printf("%f\n", output_pw16[2][2][2]);
    add_5();
    printf("%f\n", output_add5[2][2][2]);
    piecewise_17();
    printf("%f\n", output_pw17[2][2][2]);
    depthwise_9();
    printf("%f\n", output_dw9[2][2][2]);
    piecewise_18();
    printf("%f\n", output_pw18[2][2][2]);
    add_6();
    printf("%f  ", output_add6[2][2][2]);
    scale1();
    scale1();
    scale1();
    printf("%f\n", output_add6[2][2][2]);
    piecewise_19();
    printf("%f\n", output_pw19[2][2][2]);
    depthwise_10();
    printf("%f\n", output_dw10[2][2][2]);
    piecewise_20();
    printf("%f\n", output_pw20[2][2][2]);
    piecewise_21();
    printf("%f\n", output_pw21[2][2][2]);
    depthwise_11();
    printf("%f\n", output_dw11[2][2][2]);
    piecewise_22();
    printf("%f\n", output_pw22[2][2][2]);
    add_7();
    printf("%f\n", output_add7[2][2][2]);
    piecewise_23();
    printf("%f\n", output_pw23[2][2][2]);
    depthwise_12();
    printf("%f\n", output_dw12[2][2][2]);
    piecewise_24();
    printf("%f\n", output_pw24[2][2][2]);
    piecewise_25();
    printf("%f\n", output_pw25[2][2][2]);
    depthwise_13();
    printf("%f\n", output_dw13[2][2][2]);
    piecewise_26();
    printf("%f\n", output_pw26[2][2][2]);
    add_8();
    printf("%f\n", output_add8[2][2][2]);
    piecewise_27();
    printf("%f\n", output_pw27[2][2][2]);
    depthwise_14();
    printf("%f\n", output_dw14[2][2][2]);
    piecewise_28();
    printf("%f\n", output_pw28[2][2][2]);
    add_9();
    printf("%f  ", output_add9[2][2][2]);
    scale2();
    scale2();
    scale2();
    printf("%f\n", output_add9[2][2][2]);
    piecewise_29();
    printf("%f\n", output_pw29[2][2][2]);
    averagepool_0();
    printf("%f\n", output_avgpool0[2]);
    dense_0();
    printf("%f\n", output_dense0[2]);
    dense_1();
    printf("%f  ", output_dense1[2]);
    scale3();
    scale3();
    printf("%f\n", output_dense1[2]);
    softmax_0();
    printf("%f\n", output_softmax0[300]);

    printf("kkkkk\n");
    return 0;
}


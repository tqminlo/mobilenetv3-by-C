#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int extern bias_cv0[16];
int extern weight_cv0[16*3*3*3];
int extern bias_dw0[16];
int extern weight_dw0[1*3*3*16];
int extern bias_dw10[480];
int extern weight_dw10[1*3*3*480];
int extern bias_dw11[672];
int extern weight_dw11[1*3*3*672];
int extern bias_dw12[672];
int extern weight_dw12[1*3*3*672];
int extern bias_dw13[960];
int extern weight_dw13[1*3*3*960];
int extern bias_dw14[960];
int extern weight_dw14[1*3*3*960];
int extern bias_dw1[64];
int extern weight_dw1[1*3*3*64];
int extern bias_dw2[72];
int extern weight_dw2[1*3*3*72];
int extern bias_dw3[72];
int extern weight_dw3[1*3*3*72];
int extern bias_dw4[120];
int extern weight_dw4[1*3*3*120];
int extern bias_dw5[120];
int extern weight_dw5[1*3*3*120];
int extern bias_dw6[240];
int extern weight_dw6[1*3*3*240];
int extern bias_dw7[200];
int extern weight_dw7[1*3*3*200];
int extern bias_dw8[184];
int extern weight_dw8[1*3*3*184];
int extern bias_dw9[184];
int extern weight_dw9[1*3*3*184];
int extern bias_pw0[16];
int extern weight_pw0[16*1*1*16];
int extern bias_pw10[40];
int extern weight_pw10[40*1*1*120];
int extern bias_pw11[240];
int extern weight_pw11[240*1*1*40];
int extern bias_pw12[80];
int extern weight_pw12[80*1*1*240];
int extern bias_pw13[200];
int extern weight_pw13[200*1*1*80];
int extern bias_pw14[80];
int extern weight_pw14[80*1*1*200];
int extern bias_pw15[184];
int extern weight_pw15[184*1*1*80];
int extern bias_pw16[80];
int extern weight_pw16[80*1*1*184];
int extern bias_pw17[184];
int extern weight_pw17[184*1*1*80];
int extern bias_pw18[80];
int extern weight_pw18[80*1*1*184];
int extern bias_pw19[480];
int extern weight_pw19[480*1*1*80];
int extern bias_pw1[64];
int extern weight_pw1[64*1*1*16];
int extern bias_pw20[112];
int extern weight_pw20[112*1*1*480];
int extern bias_pw21[672];
int extern weight_pw21[672*1*1*112];
int extern bias_pw22[112];
int extern weight_pw22[112*1*1*672];
int extern bias_pw23[672];
int extern weight_pw23[672*1*1*112];
int extern bias_pw24[160];
int extern weight_pw24[160*1*1*672];
int extern bias_pw25[960];
int extern weight_pw25[960*1*1*160];
int extern bias_pw26[160];
int extern weight_pw26[160*1*1*960];
int extern bias_pw27[960];
int extern weight_pw27[960*1*1*160];
int extern bias_pw28[160];
int extern weight_pw28[160*1*1*960];
int extern bias_pw29[960];
int extern weight_pw29[960*1*1*160];
int extern bias_pw2[24];
int extern weight_pw2[24*1*1*64];
int extern bias_pw30[1280];
int extern weight_pw30[1280*1*1*960];
int extern bias_pw31[1001];
int extern weight_pw31[1001*1*1*1280];
int extern bias_pw3[24];
int extern weight_pw3[24*1*1*64];
int extern bias_pw4[24];
int extern weight_pw4[24*1*1*72];
int extern bias_pw5[72];
int extern weight_pw5[72*1*1*24];
int extern bias_pw6[40];
int extern weight_pw6[40*1*1*72];
int extern bias_pw7[120];
int extern weight_pw7[120*1*1*40];
int extern bias_pw8[40];
int extern weight_pw8[40*1*1*120];
int extern bias_pw9[120];
int extern weight_pw9[120*1*1*40];

int tensor0[407934] = {0};
int tensor1[407934] = {0};
int tensor2[407934] = {0};  // tensor for padding input
int tensor3[407934] = {0};  // tensor for holding to add


void padding_inp(int stride, int size, int fil, int *tensor_inp, int *tensor_pad){
    if (stride==2){
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < fil; k++) {
                    tensor_pad[i*(size+1)*fil + j*fil + k] = tensor_inp[i*size*fil + j*fil + k];
                }
            }
        }

        for (int i = 0; i < size; i++){
            for (int k = 0; k < fil; k++) {
                tensor_pad[i*(size+1)*fil + size*fil + k] = 0;
            }
        }

        for (int j = 0; j < size+1; j++){
            for (int k = 0; k < fil; k++) {
                tensor_pad[size*(size+1)*fil + j*fil + k] = 0;
            }
        }
    }

    else if (stride==1){
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < fil; k++) {
                    tensor_pad[(i+1)*(size+2)*fil + (j+1)*fil + k] = tensor_inp[i*size*fil + j*fil + k];
                }
            }
        }

        for (int i = 1; i < size+1; i++) {
            for (int k = 0; k < fil; k++) {
                tensor_pad[i*(size+2)*fil + k] = 0;
                tensor_pad[i*(size+2)*fil + (size+1)*fil + k] = 0;
            }
        }

        for (int j = 0; j < size+2; j++){
            for (int k = 0; k < fil; k++) {
                tensor_pad[j*fil + k] = 0;
                tensor_pad[(size+1)*(size+2)*fil + j*fil + k] = 0;
            }
        }
    }
}

int offset(int i, int j, int k, int maxj, int maxk){
    return i*maxj*maxk + j*maxk + k;
}

void init_input(int *tensor_inp) {
    for (int i = 0; i < 224; i++) {
        for (int j = 0; j < 224; j++) {
            for (int k = 0; k < 3; k++) {
                tensor_inp[offset(i, j, k, 224, 3)] = 1;
            }
        }
    }
}

void conv(int int_size, int out_size, int int_fil, int out_fil, int *tensor_inp, int *tensor_pad, int *tensor_out, int *kernel, int *bias){
    // padding input
    padding_inp(2, int_size, int_fil, tensor_inp, tensor_pad);

    // Conv2d: kernel = 3x3, stride = 2
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < out_fil; k++) {
                int sum = 0;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        for (int l = 0; l < 3; l++) {
                            int id_pad = offset(i*2+ii, j*2+jj, k, int_size+1, int_fil); // stride = 2
                            int id_kernel = k*3*3*3 + ii*3*3 + jj*3 + l;
                            sum += tensor_pad[id_pad] * kernel[id_kernel];
                        }
                    }
                }
                int id_out = offset(i, j, k, out_size, out_fil);
                tensor_out[id_out] = sum + bias[k]; // bias
            }
        }
    }
}

void pw(int size, int inp_fil, int out_fil, int *tensor_inp, int *tensor_out, int *kernel, int *bias){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < out_fil; k++) {
                int sum = 0;
                for (int l = 0; l < inp_fil; l++) {
                    int id_inp = offset(i, j, l, size, inp_fil); // stride = 1, kernel size = 1
                    int id_kernel = k*inp_fil + l;
                    sum += tensor_inp[id_inp] * kernel[id_kernel];  // stride = 1
                }
                int id_out = offset(i, j, k, size, out_fil);
                tensor_out[id_out] = sum + bias[k];  // bias
            }
        }
    }
}

void dw(int stride, int out_size, int fil, int *tensor_inp, int *tensor_pad, int *tensor_out, int *kernel, int *bias){
    int inp_size = out_size * stride;
    // padding input
    padding_inp(stride, inp_size, fil, tensor_inp, tensor_pad);

    // Dw: kernel_size = 3x3, stride = 1 or 2
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < fil; k++) {
                int sum = 0;
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        int id_pad = offset(i*stride+ii, j*stride+jj, k, inp_size+3-stride, fil);
                        int id_kernel = offset(ii, jj, k, 3, fil);
                        sum += tensor_pad[id_pad] * kernel[id_kernel];
                    }
                }
                int id_out = offset(i, j, k, out_size, fil);
                tensor_out[id_out] = sum + bias[k]; // bias
            }
        }
    }
}

void add(int size, int fil, int *tensor_inp1, int *tensor_inp2, int *tensor_out){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < fil; k++) {
                int id_off = offset(i, j, k, size, fil);
                tensor_out[id_off] = tensor_inp1[id_off] + tensor_inp2[id_off];
            }
        }
    }
}

void avgpool(int size, int fil, int *tensor_inp, int *tensor_out){
    for (int k = 0; k < fil; k++) {
        int sum = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++){
                int id_off = offset(i, j, k, size, fil);
                sum += tensor_inp[id_off];
            }
        }
        tensor_out[k] = sum / (size*size);
    }
}

int main(){
    init_input(tensor0);
    printf("%d\n", tensor0[1]);
    conv(224, 112, 3, 16, tensor0, tensor2, tensor3, weight_cv0, bias_cv0);
    printf("%d\n", tensor3[1]);
    dw(1, 112, 16, tensor3, tensor2, tensor0, weight_dw0, bias_dw0);
    printf("%d\n", tensor0[1]);
    pw(112, 16, 16, tensor0, tensor1, weight_pw0, bias_pw0);
    printf("%d\n", tensor1[1]);
    add(112, 16, tensor3, tensor1, tensor0);
    printf("%d\n", tensor0[1]);

    pw(112, 16, 64, tensor0, tensor1, weight_pw1, bias_pw1);
    printf("%d\n", tensor1[1]);
    dw(2, 56, 64, tensor1, tensor2, tensor0, weight_dw1, bias_dw1); //dw1
    printf("%d\n", tensor0[1]);
    pw(56, 64, 24, tensor0, tensor3, weight_pw2, bias_pw2); //pw2
    printf("%d\n", tensor3[1]);
    pw(56, 24, 72, tensor3, tensor0, weight_pw3, bias_pw3); //pw3
    printf("%d\n", tensor0[1]);
    dw(1, 56, 72, tensor0, tensor2, tensor1, weight_dw2, bias_dw2);
    printf("%d\n", tensor1[1]);
    pw(56, 72, 24, tensor1, tensor0, weight_pw4, bias_pw4); //pw4
    printf("%d\n", tensor0[1]);
    add(56, 24, tensor3, tensor0, tensor1);
    printf("%d\n", tensor1[1]);

    pw(56, 24, 72, tensor1, tensor0, weight_pw5, bias_pw5); //pw5
    dw(2, 28, 72, tensor0, tensor2, tensor1, weight_dw3, bias_dw3); //dw1
    pw(28, 72, 40, tensor1, tensor3, weight_pw6, bias_pw6); //pw6
    pw(28, 40, 120, tensor3, tensor1, weight_pw7, bias_pw7); //pw8
    dw(1, 28, 120, tensor1, tensor2, tensor0, weight_dw4, bias_dw4);
    pw(28, 120, 40, tensor0, tensor1, weight_pw8, bias_pw8); //pw8
    add(28, 40, tensor3, tensor1, tensor3);

    pw(28, 40, 120, tensor3, tensor1, weight_pw9, bias_pw9); //pw9
    dw(1, 28, 120, tensor1, tensor2, tensor0, weight_dw5, bias_dw5);
    pw(28, 120, 40, tensor0, tensor1, weight_pw10, bias_pw10); //pw10
    add(28, 40, tensor3, tensor1, tensor0);

    pw(28, 40, 240, tensor0, tensor1, weight_pw11, bias_pw11); //pw11
    dw(2, 14, 240, tensor1, tensor2, tensor0, weight_dw6, bias_dw6);
    pw(14, 240, 80, tensor0, tensor3, weight_pw12, bias_pw12); //pw12
    pw(14, 80, 200, tensor3, tensor0, weight_pw13, bias_pw13); //pw13
    dw(1, 14, 200, tensor0, tensor2, tensor1, weight_dw7, bias_dw7);
    pw(14, 200, 80, tensor1, tensor0, weight_pw14, bias_pw14); //pw14
    add(14, 80, tensor3, tensor0, tensor3);

    pw(14, 80, 184, tensor3, tensor0, weight_pw15, bias_pw15); //pw15
    dw(1, 14, 184, tensor0, tensor2, tensor1, weight_dw8, bias_dw8);
    pw(14, 184, 80, tensor1, tensor0, weight_pw16, bias_pw16); //pw16
    add(14, 80, tensor3, tensor0, tensor3);

    pw(14, 80, 184, tensor3, tensor0, weight_pw17, bias_pw17); //pw17
    dw(1, 14, 184, tensor0, tensor2, tensor1, weight_dw9, bias_dw9);
    pw(14, 184, 80, tensor1, tensor0, weight_pw18, bias_pw18); //pw18
    add(14, 80, tensor3, tensor0, tensor1);

    pw(14, 80, 480, tensor1, tensor0, weight_pw19, bias_pw19); //pw19
    dw(1, 14, 480, tensor0, tensor2, tensor1, weight_dw10, bias_dw10);
    pw(14, 480, 112, tensor1, tensor3, weight_pw20, bias_pw20); //pw20
    pw(14, 112, 672, tensor3, tensor0, weight_pw21, bias_pw21); //pw21
    dw(1, 14, 672, tensor0, tensor2, tensor1, weight_dw11, bias_dw11);
    pw(14, 672, 112, tensor1, tensor0, weight_pw22, bias_pw22); //pw22
    add(14, 112, tensor3, tensor0, tensor1);

    pw(14, 112, 672, tensor1, tensor0, weight_pw23, bias_pw23); //pw23
    dw(2, 7, 672, tensor0, tensor2, tensor1, weight_dw12, bias_dw12);
    pw(7, 672, 160, tensor1, tensor3, weight_pw24, bias_pw24); //pw24
    pw(7, 160, 960, tensor3, tensor0, weight_pw25, bias_pw25); //pw25
    dw(1, 7, 960, tensor0, tensor2, tensor1, weight_dw13, bias_dw13);
    pw(7, 960, 160, tensor1, tensor0, weight_pw26, bias_pw26); //pw26
    add(7, 160, tensor3, tensor0, tensor3);

    pw(7, 160, 960, tensor3, tensor0, weight_pw27, bias_pw27); //pw27
    dw(1, 7, 960, tensor0, tensor2, tensor1, weight_dw14, bias_dw14);
    pw(7, 960, 160, tensor1, tensor0, weight_pw28, bias_pw28); //pw28
    add(7, 160, tensor3, tensor0, tensor1);

    pw(7, 160, 960, tensor1, tensor0, weight_pw29, bias_pw29); //pw29
    avgpool(7, 960, tensor0, tensor1);
    pw(1, 960, 1280, tensor1, tensor0, weight_pw30, bias_pw30); //pw30
    avgpool(1, 1280, tensor0, tensor1);
    pw(1, 1280, 1001, tensor1, tensor0, weight_pw31, bias_pw31); //pw31
    printf("%d\n", tensor0[1]);

    printf("kkk");
}
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

int tensor0[300000] = {0};
int tensor1[300000] = {0};
int tensor2[300000] = {0};


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

int main(){
    init_input(tensor0);
    conv(224, 112, 3, 16, tensor0, tensor1, tensor2, weight_cv0, bias_cv0);
    printf("%d\n", tensor2[10]);
    dw(1, 112, 16, tensor2, tensor1, tensor0, weight_dw0, bias_dw0);
    printf("%d\n", tensor0[10]);
    pw(112, 16, 16, tensor0, tensor1, weight_pw0, bias_pw0);
    printf("%d\n", tensor1[10]);
    add(112, 16, tensor2, tensor1, tensor0);
    printf("%d\n", tensor0[10]);

}
#define NN_IMPL

#include <time.h>

#include "nn.h"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

int main(void) {
    srand(time(0));

    const size_t stride = 3;
    const size_t n = sizeof(td) / sizeof(td[0]) / stride;

    const Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };
    const Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };

    const size_t arch[] = {2, 4, 1};
    const NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    const NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    for (size_t i = 0; i < 2 * 1000 * 1000; ++i) {
        const float rate = 1e-1;
        const float eps = 1e-1;
        nn_finite_diff(nn, g, eps, ti, to);
        nn_learn(nn, g, rate);
    }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn, a_sigmoid_f);
            printf("%zu ^ %zu = %f\n", i, j, roundf(MAT_AT(NN_OUTPUT(nn), 0, 0)));
        }
    }

    return 0;
}
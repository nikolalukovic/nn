#ifndef NN_H_
#define NN_H_

#include <stddef.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

#define MAT_AT(m, i, j) (m).es[(i) * m.stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m, 0)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float rand_float(void);

float a_sigmoid_f(float x);

float a_relu_f(float x);

float a_tanh_f(float x);

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

Mat mat_alloc(size_t rows, size_t cols);

void mat_rand(Mat m, float low, float high);

Mat mat_row(Mat m, size_t row);

void mat_copy(Mat dst, Mat src);

void mat_fill(Mat a, float v);

void mat_dot(Mat dst, Mat a, Mat b);

void mat_sum(Mat dst, Mat a);

void mat_apply_af(Mat m, float (*f)(float));

void mat_print(Mat m, const char *name, size_t padding);

#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

typedef struct {
    size_t count;

    Mat *ws;
    Mat *bs;
    Mat *as; // count + 1
} NN;

NN nn_alloc(const size_t *arch, size_t arch_count);

void nn_print(NN nn, const char *name);

void nn_rand(NN nn, float low, float high);

void nn_forward(NN nn, float (*a_func)(float));

float nn_cost(NN nn, Mat ti, Mat to);

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);

void nn_learn(NN nn, NN g, float rate);
#endif

#ifdef NN_IMPL
#include <stdio.h>
#include <math.h>

inline float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

inline float a_sigmoid_f(const float x) {
    return 1.f / (1.f + expf(-x));
}

inline float a_relu_f(const float x) {
    return MAX(0, x);
}

inline float a_tanh_f(const float x) {
    return MAX(0, tanhf(x));
}

inline Mat mat_alloc(const size_t rows, const size_t cols) {
    Mat m;

    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);

    NN_ASSERT(m.es != NULL);

    return m;
}

inline void mat_rand(const Mat m, const float low, const float high) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

inline Mat mat_row(const Mat m, const size_t row) {
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}

inline void mat_copy(const Mat dst, const Mat src) {
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

inline void mat_fill(const Mat m, const float v) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = v;
        }
    }
}

inline void mat_dot(const Mat dst, const Mat a, const Mat b) {
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    size_t n = a.cols;

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

inline void mat_sum(const Mat dst, const Mat a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

inline void mat_apply_af(const Mat m, float (*f)(float)) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = f(MAT_AT(m, i, j));
        }
    }
}

inline void mat_print(const Mat m, const char *name, const size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}

inline NN nn_alloc(const size_t *arch, const size_t arch_count) {
    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);

    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);

    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i) {
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}

inline void nn_print(const NN nn, const char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
        char buf[256];
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

inline void nn_rand(const NN nn, const float low, const float high) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

inline void nn_forward(const NN nn, float (*a_func)(float)) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_apply_af(nn.as[i + 1], a_func);
    }
}

inline float nn_cost(const NN nn, const Mat ti, const Mat to) {
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);

    size_t input_rows = ti.rows;
    size_t output_rows = to.rows;

    float c = 0;
    for (size_t i = 0; i < input_rows; i++) {
        Mat input_row = mat_row(ti, i);
        Mat output_row = mat_row(to, i);

        mat_copy(NN_INPUT(nn), input_row);
        nn_forward(nn, a_sigmoid_f);

        for (size_t j = 0; j < output_rows; j++) {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(output_row, 0, j);
            c += d * d;
        }
    }

    return c / input_rows;
}

inline void nn_finite_diff(const NN nn, const NN g, const float eps, const Mat ti, const Mat to) {
    float saved;
    float c = nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

inline void nn_learn(const NN nn, const NN g, const float rate) {
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

#endif

/* symnmf.c */
#include "symnmf.h"
#include <string.h>
#include <math.h>

#define ERROR_MSG "An Error Has Occurred"
#define MAX_ITER 300
#define EPSILON 1e-4
#define BETA 0.5



double** alloc_2d_array(int n, int d) {
    double** mat = (double**)malloc(n * sizeof(double*));
    int i;
    if (!mat) return NULL;
    for (i = 0; i < n; i++) {
        mat[i] = (double*)calloc(d, sizeof(double));
        if (!mat[i]) return NULL;
    }
    return mat;
}

void free_matrix(double** mat, int n) {
    int i;
    if (!mat) return;
    for (i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

/* === I/O Operations === */

void print_matrix(double** mat, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%.4f", mat[i][j]);
            if (j < m - 1) printf(",");
        }
        printf("\n");
    }
}

int is_file_empty(const char *filename) {
    FILE *fp = fopen(filename, "r");
    int c;
    if (fp == NULL) return 1;
    c = fgetc(fp);
    fclose(fp);
    return c == EOF;
}

void get_matrix_dimensions(FILE* fp, int* n, int* d) {
    char line[2048];
    *n = 0;
    *d = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strlen(line) <= 1) continue;
        if (*n == 0) {
            char* token = strtok(line, ",");
            while (token) {
                (*d)++;
                token = strtok(NULL, ",");
            }
        }
        (*n)++;
    }
}

double** read_input_file(const char* filename, int* n, int* d) {
    FILE* fp = fopen(filename, "r");
    double** X = NULL;
    int i, j;
    if (!fp) return NULL;

    get_matrix_dimensions(fp, n, d);
    rewind(fp);
    
    X = alloc_2d_array(*n, *d);
    if (!X) {
        fclose(fp);
        return NULL;
    }
    
    for (i = 0; i < *n; i++) {
        for (j = 0; j < *d; j++) {
            if (fscanf(fp, "%lf", &X[i][j]) != 1) {
                free_matrix(X, *n);
                fclose(fp);
                return NULL;
            }
            if (j < *d - 1) fgetc(fp);
        }
        fgetc(fp);
    }
    fclose(fp);
    return X;
}

/* === Math Logic === */

double** compute_similarity_matrix(double** X, int n, int d) {
    int i, j, k;
    double diff, sum;
    double** A = alloc_2d_array(n, n);
    if (!A) return NULL;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0;
            } else {
                sum = 0;
                for (k = 0; k < d; k++) {
                    diff = X[i][k] - X[j][k];
                    sum += diff * diff;
                }
                A[i][j] = exp(-sum / 2.0);
            }
        }
    }
    return A;
}

double** compute_diagonal_degree_matrix(double** A, int n) {
    int i, j;
    double** D = alloc_2d_array(n, n);
    if (!D) return NULL;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i][i] += A[i][j];
        }
    }
    return D;
}

double** compute_normalized_similarity_matrix(double** A, double** D, int n) {
    int i, j;
    double** W = alloc_2d_array(n, n);
    double* D_inv_sqrt;
    
    if (!W) return NULL;

    D_inv_sqrt = (double*)malloc(n * sizeof(double));
    if (!D_inv_sqrt) {
        free_matrix(W, n);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        D_inv_sqrt[i] = 1.0 / sqrt(D[i][i]);
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            W[i][j] = A[i][j] * D_inv_sqrt[i] * D_inv_sqrt[j];
        }
    }

    free(D_inv_sqrt);
    return W;
}

void matmul(double** A, double** B, double** C, int n, int m, int p, int transpose_B) {
    int i, j, k;
    double b_val;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            C[i][j] = 0;
            for (k = 0; k < m; k++) {
                b_val = transpose_B ? B[j][k] : B[k][j];
                C[i][j] += A[i][k] * b_val;
            }
        }
    }
}

double frobenius_norm(double** A, double** B, int n, int k) {
    double sum = 0.0, diff;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

double** update_symnmf(double** H, double** W, int n, int k) {
    int iter, i, j;
    for (iter = 0; iter < MAX_ITER; iter++) {
        double norm;
        double** WH = alloc_2d_array(n, k);
        double** HHT = alloc_2d_array(n, n);
        double** HHTH = alloc_2d_array(n, k);
        double** H_new = alloc_2d_array(n, k);

        matmul(W, H, WH, n, n, k, 0); 
        matmul(H, H, HHT, n, k, n, 1); 
        matmul(HHT, H, HHTH, n, n, k, 0); 

        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_new[i][j] = H[i][j] * (1.0 - BETA + BETA * (WH[i][j] / HHTH[i][j]));
            }
        }

        norm = frobenius_norm(H, H_new, n, k);
        
        free_matrix(WH, n);
        free_matrix(HHT, n);
        free_matrix(HHTH, n);
        free_matrix(H, n);

        H = H_new;
        if (norm < EPSILON) break;
    }
    return H;
}

/* === Routing & Execution === */

void execute_goal(const char* goal, double** X, int n, int d) {
    double **A = NULL, **D = NULL, **W = NULL;

    A = compute_similarity_matrix(X, n, d);
    if (!A) {
        printf("%s\n", ERROR_MSG);
        return;
    }

    if (strcmp(goal, "sym") == 0) {
        print_matrix(A, n, n);
    } 
    else if (strcmp(goal, "ddg") == 0) {
        D = compute_diagonal_degree_matrix(A, n);
        print_matrix(D, n, n);
        free_matrix(D, n);
    } 
    else if (strcmp(goal, "norm") == 0) {
        D = compute_diagonal_degree_matrix(A, n);
        W = compute_normalized_similarity_matrix(A, D, n);
        print_matrix(W, n, n);
        free_matrix(D, n);
        free_matrix(W, n);
    } 
    else {
        printf("%s\n", ERROR_MSG);
    }
    free_matrix(A, n);
}

int main(int argc, char* argv[]) {
    int n, d;
    double** X;

    if (argc != 3 || is_file_empty(argv[2])) {
        printf("%s\n", ERROR_MSG);
        return 1;
    }

    X = read_input_file(argv[2], &n, &d);
    if (!X) {
        printf("%s\n", ERROR_MSG);
        return 1;
    }

    execute_goal(argv[1], X, n, d);
    free_matrix(X, n);
    return 0;
}



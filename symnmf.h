/* symnmf.h */

#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* === Memory Management === */
double** alloc_2d_array(int n, int d);
void free_matrix(double** mat, int n);

/* === Matrix Printing === */
void print_matrix(double** mat, int n, int m);

/* === File Reading === */
double** read_input_file(const char* filename, int* n, int* d);

/* === Matrix Computations === */
double** compute_similarity_matrix(double** X, int n, int d);
double** compute_diagonal_degree_matrix(double** A, int n);
double** compute_normalized_similarity_matrix(double** A, double** D, int n);

/* === Matrix Algebra === */
void matmul(double** A, double** B, double** C, int n, int m, int p, int transpose_B);
double frobenius_norm(double** A, double** B, int n, int k);

/* === SymNMF Update === */
double** update_symnmf(double** H, double** W, int n, int k);

#endif /* SYMNMF_H */

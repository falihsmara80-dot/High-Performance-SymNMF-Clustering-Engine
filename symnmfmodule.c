#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/*  Convert Python 2D List to C double** */
static double** py_list_to_c_array(PyObject* py_list, int n, int d) {
    int i, j;
    double** c_array = alloc_2d_array(n, d);
    for (i = 0; i < n; i++) {
        PyObject* row = PyList_GetItem(py_list, i);
        for (j = 0; j < d; j++) {
            c_array[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    return c_array;
}

/* Helper: Convert C double** to Python 2D List */
static PyObject* c_array_to_py_list(double** c_array, int n, int d) {
    int i, j;
    PyObject* py_list = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyObject* py_row = PyList_New(d);
        for (j = 0; j < d; j++) {
            PyList_SetItem(py_row, j, PyFloat_FromDouble(c_array[i][j]));
        }
        PyList_SetItem(py_list, i, py_row);
    }
    return py_list;
}

/* Wrapper for compute_similarity_matrix */
static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject *data_list, *result;
    int n, d;
    double **X, **A;

    if (!PyArg_ParseTuple(args, "O", &data_list)) return NULL;

    n = PyList_Size(data_list);
    d = PyList_Size(PyList_GetItem(data_list, 0));

    X = py_list_to_c_array(data_list, n, d);
    A = compute_similarity_matrix(X, n, d);
    result = c_array_to_py_list(A, n, n);

    free_matrix(X, n);
    free_matrix(A, n);
    return result;
}

/* Wrapper for compute_diagonal_degree_matrix */
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject *sim_list, *result;
    int n;
    double **A, **D;

    if (!PyArg_ParseTuple(args, "O", &sim_list)) return NULL;

    n = PyList_Size(sim_list);
    A = py_list_to_c_array(sim_list, n, n);
    D = compute_diagonal_degree_matrix(A, n);
    result = c_array_to_py_list(D, n, n);

    free_matrix(A, n);
    free_matrix(D, n);
    return result;
}

/* Wrapper for compute_normalized_similarity_matrix */
static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject *sim_list, *ddg_list, *result;
    int n;
    double **A, **D, **W;

    if (!PyArg_ParseTuple(args, "OO", &sim_list, &ddg_list)) return NULL;

    n = PyList_Size(sim_list);
    A = py_list_to_c_array(sim_list, n, n);
    D = py_list_to_c_array(ddg_list, n, n);
    W = compute_normalized_similarity_matrix(A, D, n);
    result = c_array_to_py_list(W, n, n);

    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);
    return result;
}

/* Wrapper for update_symnmf */
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *h_list, *w_list, *result;
    int n, k;
    double **H, **W, **final_H;

    if (!PyArg_ParseTuple(args, "OO", &h_list, &w_list)) return NULL;

    n = PyList_Size(h_list);
    k = PyList_Size(PyList_GetItem(h_list, 0));

    H = py_list_to_c_array(h_list, n, k);
    W = py_list_to_c_array(w_list, n, n);
    
    final_H = update_symnmf(H, W, n, k);
    result = c_array_to_py_list(final_H, n, k);

    free_matrix(W, n);
    free_matrix(final_H, n); 
    return result;
}

/* Method definitions */
static PyMethodDef SymnmfMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Run SymNMF optimization"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymnmfMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
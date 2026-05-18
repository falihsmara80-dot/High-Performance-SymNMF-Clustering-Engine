import sys
import os
import numpy as np
import symnmfmodule

ERROR_MSG = "An Error Has Occurred"
np.random.seed(1234)

def exit_with_error():
    print(ERROR_MSG)
    sys.exit(1)

def validate_arguments():
    if len(sys.argv) != 4:
        exit_with_error()
    try:
        k = int(sys.argv[1])
        return k, sys.argv[2], sys.argv[3]
    except ValueError:
        exit_with_error()

def load_input_data(filename):
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        exit_with_error()
    try:
        data = np.loadtxt(filename, delimiter=',', ndmin=2).tolist()
        if not data or not all(isinstance(row, list) for row in data):
            raise ValueError
        return data
    except Exception:
        exit_with_error()

def print_matrix(matrix):
    for row in matrix:
        print(','.join(f"{x:.4f}" for x in row))

def run_symnmf(data, k):

    if k < 2 or k >= len(data):
        exit_with_error()
    
    A = symnmfmodule.sym(data)
    D = symnmfmodule.ddg(A)
    W = symnmfmodule.norm(A, D)
    
    upper = 2 * (np.mean(W) / k) ** 0.5
    H0 = np.random.uniform(0, upper, size=(len(data), k)).tolist()
    
    return symnmfmodule.symnmf(H0, W)

def main():
    k, target, input_file = validate_arguments()
    data = load_input_data(input_file)

    
    if k >= len(data) or k < 0:
        exit_with_error()

    if target == "symnmf":
        print_matrix(run_symnmf(data, k))
    elif target == "sym":
        print_matrix(symnmfmodule.sym(data))
    elif target == "ddg":
        A = symnmfmodule.sym(data)
        print_matrix(symnmfmodule.ddg(A))
    elif target == "norm":
        A = symnmfmodule.sym(data)
        D = symnmfmodule.ddg(A)
        print_matrix(symnmfmodule.norm(A, D))
    else:
        exit_with_error()

if __name__ == "__main__":
    main()
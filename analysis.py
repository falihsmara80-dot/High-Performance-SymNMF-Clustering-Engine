import sys
import os
import numpy as np
from sklearn.metrics import silhouette_score
from symnmf import run_symnmf, exit_with_error
from kmeans import get_kmeans_clustering

np.random.seed(1234)

def validate_arguments():
    if len(sys.argv) != 3:
        exit_with_error()
    try:
        k = int(sys.argv[1])
        return k, sys.argv[2]
    except ValueError:
        exit_with_error()

def validate_input_file(filename):
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        exit_with_error()
    try:
        data = np.loadtxt(filename, delimiter=',', ndmin=2).tolist()
        if not data:
            raise ValueError
        return data
    except Exception:
        exit_with_error()

def main():
    k, input_file = validate_arguments()
    X = validate_input_file(input_file)
    n = len(X)

    if k <= 1 or k >= n:
        exit_with_error()

    final_H = run_symnmf(X, k)

    symnmf_clusters = [np.argmax(row) for row in final_H]
    kmeans_clusters = get_kmeans_clustering(k, 300, X)

    print("nmf:", '%.4f' % silhouette_score(X, symnmf_clusters))
    print("kmeans:", '%.4f' % silhouette_score(X, kmeans_clusters))

if __name__ == "__main__":
    main()
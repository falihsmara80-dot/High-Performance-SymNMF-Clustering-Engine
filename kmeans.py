import re
import math
import sys

def validate_natural_variables(number):
    try:
        val = float(number)
        return val.is_integer()
    except ValueError:
        return False

def euclidean_distance(num_1, num_2):
    distance = 0
    assert len(num_1) == len(num_2)
    for i in range(len(num_1)):
        distance += (num_1[i] - num_2[i]) ** 2
    return math.sqrt(distance)

def create_new_centroid(cluster):
    list_per_coordinate = zip(*cluster)
    new_centroid = [sum(coordinate) / len(cluster) for coordinate in list_per_coordinate]
    return new_centroid

def validate_inputfile(input_data):
    if not input_data.lower().endswith('.txt'):
        print("NA")
        sys.exit(1)

def validate_and_parse_input(K, iter, input_data):
    validate_inputfile(input_data)
    try:
        with open(input_data, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        data = [[float(x) for x in re.split(r',', line)] for line in lines]
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

    if K >= len(data) or K <= 1:
        print("Invalid number of clusters!")
        sys.exit(1)


    if iter <= 1 or iter >= 1000:
        print("Invalid maximum iteration!")
        sys.exit(1)

    return data

def create_cluster(K, iter, data):
    centroids = data[:K]
    for _ in range(iter):
        clusters = {i: [] for i in range(K)}
        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            clusters[distances.index(min(distances))].append(point)
        
        new_centroids = [create_new_centroid(clusters[i]) for i in range(K)]
        
        if all(euclidean_distance(c, nc) < 0.0001 for c, nc in zip(centroids, new_centroids)):
            break
        centroids = new_centroids
    return centroids

def get_kmeans_clustering(K, iter, data):
    centroids = create_cluster(K, iter, data)
    labels = []
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        labels.append(distances.index(min(distances)))
    return labels

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        K = int(sys.argv[1])
    except:
        print("Invalid number of clusters!")
        sys.exit(1)
        
    if not validate_natural_variables(K):
        print("Invalid number of clusters!")
        sys.exit(1)
        
    try:
        iter = int(sys.argv[2]) if len(sys.argv) == 4 else 200
    except:
        print("Invalid maximum iteration!")
        sys.exit(1)
        
    if not validate_natural_variables(iter):
        print("Invalid maximum iteration!")
        sys.exit(1)
        
    data = validate_and_parse_input(K, iter, sys.argv[-1])
    
    try:
        centroids = create_cluster(K, iter, data)
        for centroid in centroids:
            print(','.join(f"{coordinate:.4f}" for coordinate in centroid))
    except:
        print("An Error Has Occurred")
        sys.exit(1)
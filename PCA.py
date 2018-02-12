import numpy as np
def pca(data):
    dvect = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j]: dvect.append([i, j])
    dvect = np.array(dvect, dtype=np.float32)
    dvect = np.array(dvect) - np.mean(dvect, axis=0)
    dvect /= np.std(dvect, 0)
    # dvect -= np.mean(dvect,axis=0)
    cov = np.dot(dvect.T, dvect) / dvect.shape[0]
    eigenval, eigenvect = np.linalg.eigh(cov)
    return cov, eigenvect, eigenval
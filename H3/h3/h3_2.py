import numpy as np

design= np.matrix('1 0.8 0.64 0.512; 1 1 1 1; 1 1.2 1.44 1.728; 1 1.4 1.96 2.744; 1 1.6 2.56 4.096')
identity = np.matrix('2 0 0 0; 0 2 0 0; 0 0 2 0; 0 0 0 2')
target = np.matrix('24; 20; 10; 13; 12')

a = np.linalg.pinv(np.add(np.matmul(np.matrix.transpose(design), design), identity))
b = np.matmul(a, np.matrix.transpose(design))
w = np.matmul(b, target)
print(w)

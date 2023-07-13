from scipy.stats import multivariate_normal
from numpy import multiply
from numpy import add
from numpy import transpose
from numpy import array
from numpy import newaxis

# I. 1)

x1 = array([1,2])
x2 = array([-1,1])
x3 = array([1,0])

u1 = array([2,2])
u2 = array([0,0])

cvm1 = array([[2,1],[1,2]])
cvm2 = array([[2,0],[0,2]])

pc1 = multivariate_normal(mean=u1, cov=cvm1)
pc2 = multivariate_normal(mean=u2, cov=cvm2)

print("p(xn, ck=1)")
print(round(pc1.pdf(x1), 5) / 2)
print(round(pc1.pdf(x2), 5) / 2)
print(round(pc1.pdf(x3), 5) / 2)
print(round(pc2.pdf(x1), 5) / 2)
print(round(pc2.pdf(x2), 5) / 2)
print(round(pc2.pdf(x3), 5) / 2)

px1 = (round(pc1.pdf(x1), 5) / 2) + (round(pc2.pdf(x1), 5) / 2)
px2 = (round(pc1.pdf(x2), 5) / 2) + (round(pc2.pdf(x2), 5) / 2)
px3 = (round(pc1.pdf(x3), 5) / 2) + (round(pc2.pdf(x3), 5) / 2)

print("pxn")
print(px1)
print(px2)
print(px3)

gamac11 = (round(pc1.pdf(x1), 5) / 2) / px1
gamac21 = (round(pc1.pdf(x2), 5) / 2) / px2
gamac31 = (round(pc1.pdf(x3), 5) / 2) / px3
gamac12 = (round(pc2.pdf(x1), 5) / 2) / px1
gamac22 = (round(pc2.pdf(x2), 5) / 2) / px2
gamac32 = (round(pc2.pdf(x3), 5) / 2) / px3

print("gama")
print(gamac11)
print(gamac21)
print(gamac31)
print(gamac12)
print(gamac22)
print(gamac32)

print("N")
N1 = gamac11+gamac21+gamac31
N2 = gamac12+gamac22+gamac32

print(N1)
print(N2)

nu1 = array(multiply((1/N1), add(multiply(gamac11, x1), multiply(gamac21, x2), multiply(gamac31, x3))))
nu2 = array(multiply((1/N2), add(multiply(gamac12, x1), multiply(gamac22, x2), multiply(gamac32, x3))))

print("new mean vectors")
print(nu1)
print(nu2)

# Allows for correct multidimensional matrix operations
x1 = x1[newaxis].T
x2 = x2[newaxis].T
x3 = x3[newaxis].T
nu1 = nu1[newaxis].T
nu2 = nu2[newaxis].T

ncvm1 = multiply((1/N1) , add(multiply(gamac11, multiply(add(x1, multiply(-1, nu1)), add(x1, multiply(-1, nu1)).T)), \
multiply(gamac21, multiply(add(x2, multiply(-1, nu1)), add(x2, multiply(-1, nu1)).T)),\
multiply(gamac31, multiply(add(x3, multiply(-1, nu1)), add(x3, multiply(-1, nu1)).T))))

ncvm2 = multiply((1/N2) , add(multiply(gamac12, multiply(add(x1, multiply(-1, nu2)), add(x1, multiply(-1, nu2)).T)), \
multiply(gamac22, multiply(add(x2, multiply(-1, nu2)), add(x2, multiply(-1, nu2)).T)),\
multiply(gamac32, multiply(add(x3, multiply(-1, nu2)), add(x3, multiply(-1, nu2)).T))))

print("new covariance matrices")
print(ncvm1)
print(ncvm2)

np1 = N1 / 3
np2 = N2 / 3

print("np")
print(np1)
print(np2)

# I. 2)


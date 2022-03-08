import numpy.linalg as linalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('regressiondata.csv')

x = data['x']
y = data['y']
x = x.sort_values()
y = y.sort_values()


def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    return X


def pol_regression(features_train, y_train, degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    polynomial_coefficients = np.linalg.solve(XX, X.transpose().dot(y_train))
    return polynomial_coefficients


plt.figure()
plt.plot(x,y, 'g')


w1 = pol_regression(x,y,1)
Xtest1 = getPolynomialDataMatrix(x, 1)
ytest1 = Xtest1.dot(w1)
plt.plot(x, ytest1, 'r')

w2 = pol_regression(x,y,2)
Xtest2 = getPolynomialDataMatrix(x, 2)
ytest2 = Xtest2.dot(w2)
plt.plot(x, ytest2, 'g')

w3 = pol_regression(x,y,3)
Xtest3 = getPolynomialDataMatrix(x, 3)
ytest3 = Xtest3.dot(w3)
plt.plot(x, ytest3, 'm')

w4 = pol_regression(x,y,6)
Xtest4 = getPolynomialDataMatrix(x, 6)
ytest4 = Xtest4.dot(w4)
plt.plot(x, ytest4, 'c')

w5 = pol_regression(x,y,10)
Xtest5 = getPolynomialDataMatrix(x, 10)
ytest5 = Xtest5.dot(w5)
plt.plot(x, ytest5, 'b')



plt.legend(('training points', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^10$'), loc = 'lower right')

         
plt.savefig('polynomial.png')


error1 = y-ytest1
SSE1 = error1.dot(error1)

error2 = y-ytest2
SSE2 = error2.dot(error2)

error3 = y-ytest3
SSE3 = error3.dot(error3)

error4 = y-ytest4
SSE4 = error4.dot(error4)

SSE1, SSE2, SSE3, SSE4
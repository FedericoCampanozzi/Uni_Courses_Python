import numpy as np

x = np.array([1,2,3,4], dtype=float)
print ("\n" + str(x))
b = x[np.newaxis, :]
print (b)
c = x[:, np.newaxis]
print ("\n" + str(c))

a = np.array([[1,2,3], [4,5,6]])
b = np.array([10,20,30])
err = False
print ("\n" + str(a + b))
if(err): print (a + c)
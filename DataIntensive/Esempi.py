import numpy as np

# np.newaxis

x = np.array([1,2,3,4], dtype=float)
print ("\n" + str(x))
b = x[np.newaxis, :]
print (b)
c = x[:, np.newaxis]
print ("\n" + str(c))

# broadcasting

a = np.array([[1,2,3], [4,5,6]])
b = np.array([10,20,30])
err = False
print ("\n" + str(a + b))
if(err): print (a + c)

# argsort().argsort() su array e matrici

x1 = np.array([30, 2, 10])
x1_arg = x1.argsort()

print("\nx1 : \n" + str(x1))
print("\nx1.argsort() : \n" + str(x1_arg) + " -> " + str(x1[x1_arg]))
print("\nx1.argsort().argsort() : \n" + 
      str(x1_arg.argsort()) + " -> " + 
      str(x1_arg[x1_arg.argsort()])  + " -> " +
      str(x1[x1_arg.argsort()]))


axis = 1
x2 = np.array([[30, 2, 10], [3, 15, 14] , [78, 15, 12]])
x2_arg = x2.argsort(axis)

print("\nx2 : \n" + str(x2))
print("\nx2.argsort() : \n" + str(x2_arg))
print("\nx2.argsort().argsort() : \n" + str(x2_arg.argsort(axis)))


# slicing su matrici tri-dimensionali e selezione indici

x3 = np.array([
                [[4,2,3],[1,2,3],[1,2,3]],
                [[4,1,3],[5,1,3],[5,2,3]]
              ])

print("x3 : \n" + str(x3))
print("dim : " + str(x3.ndim))
print("shape : " + str(x3.shape))
print("slicing : \n" + str(x3[1,:2,2:]))
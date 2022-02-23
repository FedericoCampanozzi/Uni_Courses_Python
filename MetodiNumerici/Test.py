# La mia libreria
import Lib
# Algoritmi Prof
import Alg_ApprossimazioneMinimiQuadrati as MQ
import Alg_IntegrazioneNumerica as INT
import Alg_InterpolazionePolinomiale as IPOL
import Alg_SistemiLineari as SIS
import Alg_ZeriDiFunzione as ZERIF
# Python library
import numpy as np
#import math
import sympy as sym
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
#Altro
import random

def PrintArray(array):
    if(len(array) == 0) :
     return "[]"
    s = "["
    for i in range(0, len(array)):
        s = s + str(array[i]) + ","
    s  = s[:-1]
    s = s + "]"
    return s

def TestMetodoDiBisezione() : 
    print("\nTEST -- TestMetodoDiBisezione")
    z = np.linspace(a,b,100)
    plt.plot(z, 0*z, 'y-')
    plt.plot(z, f_numerica(z), 'r-')
    x1, it1, xit1 = ZERIF.bisez(f_numerica, a, b, tolx)
    it2, xit2 = Lib.ZeriDiFunzioni.MetodoDiBisezione(f_numerica,  a, b, tolx) 
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Zero trovato : Prof: " + str(x1) + " IO: " + str(xit2[-1]))
    plt.plot(xit2[-1], f_numerica(0), 'bo')
    plt.show()
    print("FINE ------------------------")
    
def TestMetodoDiFalsaPosizione() : 
    print("\nTEST -- TestMetodoDiFalsaPosizione")
    z = np.linspace(a,b,100)
    plt.plot(z, 0*z, 'y-')
    plt.plot(z, f_numerica(z), 'r-')
    x1, it1, xit1 = ZERIF.regula_falsi(f_numerica, a, b, tolx, maxIt)
    it2, xit2 = Lib.ZeriDiFunzioni.MetodoDiFalsaPosizione(f_numerica,  a, b, tolx, maxIt)    
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Zero trovato : Prof: " + str(x1) + " IO: " + str(xit2[-1]))    
    plt.plot(x1, f_numerica(0), 'mo')
    plt.plot(xit2[-1], f_numerica(0), 'bo')
    plt.show()
    print("FINE ------------------------")
    
def TestMetodoDelleCorde() : 
    print("\nTEST -- TestMetodoDelleCorde")
    z = np.linspace(a,b,100)
    plt.plot(z, 0*z, 'y-')
    plt.plot(z, f_numerica(z), 'r-')
    x1, it1, xit1 = ZERIF.corde(f_numerica, f_prima_numerica, x0, tolx, tolf, maxIt)
    it2, xit2 = Lib.ZeriDiFunzioni.MetodoDelleCorde(f_numerica, f_prima_numerica, x0, maxIt, tolx, tolf)    
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Zero trovato : Prof: " + str(x1) + " IO: " + str(xit2[-1]))   
    plt.plot(x1, f_numerica(0), 'mo')
    plt.plot(xit2[-1], f_numerica(0), 'bo')
    plt.show()
    print("FINE ------------------------")

def TestMetodoDelleSecanti() : 
    print("\nTEST -- TestMetodoDelleSecanti")
    z = np.linspace(a,b,100)
    plt.plot(z, 0*z, 'y-')
    plt.plot(z, f_numerica(z), 'r-')
    x1, it1, xit1 = ZERIF.secanti(f_numerica, xm1, x0, tolx, tolf, maxIt)
    it2, xit2 = Lib.ZeriDiFunzioni.MetodoDelleSecanti(f_numerica, xm1, x0, maxIt, tolx, tolf)    
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Zero trovato : Prof: " + str(x1) + " IO: " + str(xit2[-1]))   
    plt.plot(x1, f_numerica(0), 'mo')
    plt.plot(xit2[-1], f_numerica(0), 'bo')
    plt.show()
    print("FINE ------------------------")

def TestMetodoDiNewton() : 
    print("\nTEST -- TestMetodoDiNewton")
    z = np.linspace(a,b,100)
    plt.plot(z, 0*z, 'y-')
    plt.plot(z, f_numerica(z), 'r-')
    x1, it1, xit1 = ZERIF.newton(f_numerica, f_prima_numerica, x0, tolx, tolx, maxIt)
    it2, xit2 = Lib.ZeriDiFunzioni.MetodoDiNewton(f_numerica, f_prima_numerica, x0, tolx, tolx, maxIt)    
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Zero trovato : Prof: " + str(x1) + " IO: " + str(xit2[-1]))   
    plt.plot(x1, f_numerica(0), 'mo')
    plt.plot(xit2[-1], f_numerica(0), 'bo')
    plt.show()
    print("FINE ------------------------")
    
def TestMetodoDiNewtonModificato() : 
    print("\nTEST -- TestMetodoDiNewtonModificato")
    z = np.linspace(a,b,100)
    plt.plot(z, 0*z, 'y-')
    plt.plot(z, f_numerica(z), 'r-')
    x1, it1, xit1 = ZERIF.newton_m(f_numerica, f_prima_numerica, x0, m, tolx, tolx, maxIt)
    it2, xit2 = Lib.ZeriDiFunzioni.MetodoDiNewtonModificato(f_numerica, f_prima_numerica, x0, m, tolx, tolx, maxIt)    
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Zero trovato : Prof: " + str(x1) + " IO: " + str(xit2[-1]))   
    plt.plot(x1, f_numerica(0), 'mo')
    plt.plot(xit2[-1], f_numerica(0), 'bo')
    plt.show()
    print("FINE ------------------------")
    
def TestMetodoIterativoDiPuntoFisso() :
    print("\nTEST -- TestMetodoIterativoDiPuntoFisso")
    z = np.linspace(a, b, 100)
    plt.plot(z, 0*z, 'y-')
    plt.plot(z, f_numerica(z), 'r-')
    plt.plot(z, g_numerica(z), 'b-')
    x1, it1, xit1 = ZERIF.iterazione(g_numerica, x0, tolx, maxIt)
    it2, xit2 = Lib.ZeriDiFunzioni.MetodoIterativoDiPuntoFisso(g_numerica, x0, tolx, maxIt)    
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Zero trovato : Prof: " + str(x1) + " IO: " + str(xit2[-1]))   
    plt.plot(x1, g_numerica(0), 'mo')
    plt.plot(xit2[-1], g_numerica(0), 'go')
    print("FINE ------------------------")
    return
    
def TestSolveSistemiLineari(): 
    print("\nTEST -- TestSolveSistemiLineari")
    P, L, U, flag = SIS.LU_nopivot(A)
    if flag == 0:
        x_nopivot,flag=SIS.LUsolve(L,U,P,b)
    else:
        P, L, U, flag = SIS.LU_pivot(A)
    sol1, cond1 = SIS.LUsolve(L, U, P, b)
    sol2, cond2 = Lib.SistemiLineari.Solve(A, b)
    print("---- Soluzione \nProf: " + PrintArray(sol1) + "\nMia:  " + PrintArray(sol2))
    print("FINE ------------------------")
    
def TestMetodoQRLS():
    print("\nTEST -- TestMetodoQRLS")
    print("XPoint: " + PrintArray(xPoint) + "\nYPoint: " + PrintArray(yPoint))
    sol1 = Lib.ApprossimazioneMinimiQuadrati.MetodoQRLS(xPoint, yPoint, n)
    sol2 = MQ.metodoQR(xPoint, yPoint, n)
    print("---- Soluzione \nProf: " + PrintArray(sol1) + "\nMia:  " + PrintArray(sol2))
    xmin = np.min(xPoint)
    xmax = np.max(xPoint)
    xval = np.linspace(xmin, xmax, 100)
    p = np.polyval(sol1, xval)
    plt.plot(xval, p,'r-')
    plt.plot(xPoint, yPoint, 'o')
    plt.legend(['Polinomio di approssimazione di grado ' + str(n), 'Dati'])
    plt.show()
    print("FINE ------------------------")
    
def TestInterpolazionePolinomiale():
    print("\nTEST -- TestInterpolazionePolinomiale")
    T = np.array([-55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65])
    L = np.array([3.7, 3.7, 3.52, 3.27, 3.2, 3.15, 3.15, 3.25, 3.47, 3.52, 3.65, 3.67, 3.52]) 
    xx = np.linspace(np.min(T), np.max(T),200)
    pol = IPOL.InterpL(T, L, xx)    
    pol42 = IPOL.InterpL(T, L, np.array([42]))
    pol_42 = IPOL.InterpL(T, L, np.array([-42]))    
    plt.plot(xx, pol, 'b--')
    plt.plot(T, L, 'r*')
    plt.plot(42, pol42, 'og')
    plt.plot(-42, pol_42, 'mx')
    plt.legend(['Interpolante di Lagrange', 'Punti di Interpolazione', 'Stima 1', 'Stima 2'])
    plt.show()
    print("FINE ------------------------")
    
def TestTrapezioCompositaWithTollerance():
    print("\nTEST -- TestTrapezioCompositaWithTollerance")
    approx1, it1 = INT.traptoll(f_numerica, a, b, tol)
    approx2, it2 = Lib.IntegrazioneNumerica.TrapezioCompositaWithTollerance(f_numerica, a, b, tol)
    z = np.linspace(a, b, 100)
    plt.plot(z, f_numerica(z), 'r-')
    plt.legend(['Integrale definito - Formula del Trapezio'])
    plt.fill_between(z, f_numerica(z))
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Integrale : Prof: " + str(approx1) + " IO: " + str(approx2))
    print("FINE ------------------------")
    plt.show()

def TestSimpsonCompositaWithTollerance():
    print("\nTEST -- TestSimpsonCompositaWithTollerance")
    approx1, it1 = INT.simptoll(f_numerica, a, b, tol)
    approx2, it2 = Lib.IntegrazioneNumerica.SimpsonCompositaWithTollerance(f_numerica, a, b, tol)
    z = np.linspace(a, b, 100)
    plt.plot(z, f_numerica(z), 'r-')
    plt.legend(['Integrale definito - Formula di Simpson'])
    plt.fill_between(z, f_numerica(z))
    print("---- Iterazioni fatte : Prof: " + str(it1) + " IO: " + str(it2))
    print("---- Integrale : Prof: " + str(approx1) + " IO: " + str(approx2))
    print("FINE ------------------------")
    plt.show()

testZeriDiFunzione = False
testSistemiLineari = False
testInterpolazionePolinomiale = True
testIntegrazioneNumerica = False

if testZeriDiFunzione :
    x = sym.symbols('x')
    f = sym.sqrt(x)-x**2/4
    a = 1
    b = 3
    tolx = 1e-12
    tolf = 1e-12
    maxIt = 1024
    x0 = 6.5
    xm1 = 7.5
    fprima = sym.diff(f, x, 1)
    f_numerica = lambdify(x, f, np)
    f_prima_numerica = lambdify(x, fprima, np)
    m = 0.5
    
    TestMetodoDiBisezione()
    TestMetodoDiFalsaPosizione()
    TestMetodoDelleCorde()
    TestMetodoDelleSecanti()
    TestMetodoDiNewton()
    TestMetodoDiNewtonModificato()
    
    f =  x**3 + 4*x**2 - 10
    f_numerica = lambdify(x, f, np)
    g = 1/2 * sym.sqrt(10-x**3)
    g_numerica = lambdify(x, g, np)
    
    a = 0.0
    b = 1.6
    x0 = 1.5
    TestMetodoIterativoDiPuntoFisso()
    
elif testSistemiLineari :
    A = np.array([[1,2,3],[0,0,1],[1,3,5]],dtype=float)
    b = np.array([[6],[1],[9]],dtype=float)
    
    TestSolveSistemiLineari()

elif testInterpolazionePolinomiale :
    xPoint = []
    yPoint = []
    n = random.randint(5, 8)
    nPoints = random.randint(10, 20)
    for i in range(0, nPoints) :
        xPoint.append(random.randint(-100, 100) / 10)
        yPoint.append(random.randint(-100, 100) / 10)
    
    xPoint = np.array(xPoint)
    yPoint = np.array(yPoint)

    TestMetodoQRLS()
    TestInterpolazionePolinomiale()
    
elif testIntegrazioneNumerica :
    x = sym.symbols('x')
    f = 5*x**3+x**4
    a = -2
    b = 3
    tol = 0.001
    f_numerica = lambdify(x,f,np)
    TestTrapezioCompositaWithTollerance()
    TestSimpsonCompositaWithTollerance()
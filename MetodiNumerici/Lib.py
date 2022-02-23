import numpy as np
import math
import scipy.linalg as spl

class ZeriDiFunzioni :    
    """
    Calcolo l'ordine di convergenza di un metodo
    INPUT:
        xit : Valore della x all'i-esima iterata
        nit : Numero di iterate
    OUTPUT:
        L'ordine di convergenza
    """
    @staticmethod
    def OrdineDiConvergenza(xit, nit):
      p = []
      for k in range(nit - 3):
         p.append(np.log(abs(xit[k + 2] - xit[k + 3]) / abs(xit[k + 1] - xit[k + 2])) / 
                  np.log(abs(xit[k + 1] - xit[k + 2])/ abs(xit[k] - xit[k + 1])))
      return p[-1]

    """
    Calcola il segno di un numero
    INPUT:
        x : Numero
    OUTPUT:
        0 se x ==0, 1 se x > 1, -1 se x < -1
    """
    @staticmethod
    def Sign(x):        
        if(x == 0) : return 0
        if(x > 1) : return 1
        if(x < 0) : return -1
    
    """
    Applica il metodo di bisezione per il calcolo degli zeri
    INPUT:
        F : Funzione considerata
        a : Estremo inferiore dell'intervallo considerato
        b : Estremo superiore dell'intervallo considerato
        tol : Tolleranza
    OUTPUT:
        it : Numero di iterate fatte
        xit : Le ordinate all'i-esima iterata
    """
    @staticmethod
    def MetodoDiBisezione(F, a, b, tol):
        
        EPS = np.spacing(1)
        fa = F(a)
        fb = F(b)
            
        if ZeriDiFunzioni.Sign(fa) ==  ZeriDiFunzioni.Sign(fb):
           print('MetodoDiBisezione -- Sign(f(a)) == Sign(f(b)) ---> Metodo Non Applicabile')
           return 0, []
        else:
            
            # Nel Metodo di bisezione il numero massimo di iterazioni e' calcolabile come nE >= 3.3 log10 ( (b-1)/E )
            nMaxIt = int(math.ceil(math.log((b-a)/tol)/math.log(2)))
            
            # contiete le ordinate all'i-esima iterata
            xit = []
            # iterata corrente
            it = 0
            
            while it < nMaxIt and  abs(b - a) >= tol + EPS * max( abs(a), abs(b) ):
                # (a + b)/2
                c = a + (b - a) * 0.5 # formula stabile per il calcolo del punto medio dell'intervallo
                xit.append(c) 
                
                fc = F(c)
                
                if fc == 0:
                    break
                elif ZeriDiFunzioni.Sign(fc) == ZeriDiFunzioni.Sign(fa): 
                    # Restringo l'intervallo t.c [c, b] tc Sign(f(c)) !=  Sign(f(b))
                    # Per il teorema degli zeri lo zero della funzione di trova nel sotto intervallo [c, b]
                    a = c
                    fa = fc
                elif ZeriDiFunzioni.Sign(fc) == ZeriDiFunzioni.Sign(fb):
                    # Restringo l'intervallo t.c [a, c] tc Sign(f(a)) !=  Sign(f(c))
                    # Per il teorema degli zeri lo zero della funzione di trova nel sotto intervallo [a, c]
                    b = c
                    fb = fc
                    
                it += 1
            
            return it, xit
     
    """
    Applico il metodo di falsa posizione o regula falsi, 
    questo metodo non e' altro che il miglioramento del metodo di bisezione
    INPUT:
        F : Funzione considerata
        a : Estremo inferiore dell'intervallo considerato
        b : Estremo superiore dell'intervallo considerato
        tol : Tolleranza
    OUTPUT:
        it : Numero di iterate fatte
        xit : Le ordinate all'i-esima iterata
    """
    @staticmethod
    def MetodoDiFalsaPosizione(F, a, b, tol, nMaxIt):            
        
        EPS = np.spacing(1)
        fa = F(a)
        fb = F(b)
            
        if ZeriDiFunzioni.Sign(fa) ==  ZeriDiFunzioni.Sign(fb):
           print('MetodoDiFalsaPosizione -- Sign(f(a)) == Sign(f(b)) ---> Metodo Non Applicabile')
           return 0, []
        else:
            # contiete le ordinate all'i-esima iterata
            xit = []
            # iterata corrente
            it = 0
            
            while it < nMaxIt and  abs(b-a) >= tol + EPS * max(abs(a), abs(b)) and abs(fa) >= tol :
                c = a - fa * ( (b - a) / (fb - fa))
                xit.append(c) 
                
                fc = F(c)
                
                if fc == 0:
                    break
                elif ZeriDiFunzioni.Sign(fc) == ZeriDiFunzioni.Sign(fa): 
                    # Restringo l'intervallo t.c [c, b] tc Sign(f(c)) !=  Sign(f(b))
                    # Per il teorema degli zeri lo zero della funzione di trova nel sotto intervallo [c, b]
                    a = c
                    fa = fc
                elif ZeriDiFunzioni.Sign(fc) == ZeriDiFunzioni.Sign(fb):
                    # Restringo l'intervallo t.c [a, c] tc Sign(f(a)) !=  Sign(f(c))
                    # Per il teorema degli zeri lo zero della funzione di trova nel sotto intervallo [a, c]
                    b = c
                    fb = fc
                    
                it += 1
            
            if(it >= nMaxIt) :
                print('MetodoDiFalsaPosizione -- Numero massimo di iterazioni raggiunto')
            
            return it, xit
    
    """
    Applico il metodo delle corde quindi abbiamo mi = costante
    INPUT:
       F : Funzione considerata
       DF : Derivata prima della F
       x0 : Il valore di innesco
       nMaxIt : Numero massimo di iterazioni
       tolf : Tolleranza sulla funzione
       tolx : Tolleranza sul valore della x
    OUTPUT:
       it : Numero di iterate fatte
       xit : Le ordinate all'i-esima iterata
    """
    @staticmethod
    def MetodoDelleCorde(F, DF, x0, nMaxIt, tolf, tolx) :
       
       xit=[]
       
       # Coefficiente angolare della tangente in x0

       m = DF(x0)
       
       fx0 = F(x0)       
       d = fx0 / m
       x1 = x0 - d
       
       fx1 = F(x1)
       xit.append(x1)
       
       it = 1
       
       while it < nMaxIt and  abs(fx1) >= tolf and abs(d) >= tolx * abs(x1) :
           x0 = x1
           fx0 = F(x0)
           d = fx0/m
           
           # x1 = ascissa del punto di intersezione tra  la retta che passa per il punto 
           # (xi,f(xi)) e ha pendenza uguale a m  e l'asse x
           
           x1 = x0 - d
           fx1 = F(x1)
           
           xit.append(x1)
           it += 1
          
       if it >= nMaxIt:
           print('MetodoDelleCorde -- Numero massimo di iterazioni raggiunto')
            
       return it, xit

    """
    Applico il metodo delle secandi quindi abbiamo dalla teoria che
    mi = (F(xi) - F(xi-1))/(xi - xi-1) ----> xi+1 = xi - f(xi)((xi-xi-1)  / (f(xi) - f(xi-1)))
    INPUT:
       nMaxIt : Numero massimo di iterazioni
    OUTPUT:
       it : Numero di iterate fatte
       xit : Le ordinate all'i-esima iterata
    """
    @staticmethod
    def MetodoDelleSecanti(F, xm1, x0, nMaxIt, tolf, tolx) :
        xit = []
        fxm1 = F(xm1)
        fx0 = F(x0)
        d = (fx0 * (x0 - xm1) / (fx0 - fxm1))
        x1 = x0 - d        
        xit.append(x1)
        fx1 = F(x1);
        it = 1
       
        while it < nMaxIt and abs(fx1) >=  tolf and abs(d) >=  tolx * abs(x1):
           xm1 = x0
           x0 = x1
           fxm1 = F(xm1)
           fx0 = F(x0) 
           d = fx0 * (x0 - xm1) / (fx0 - fxm1)
           x1 = x0 - d
           fx1 = F(x1)
           xit.append(x1)
           it += 1
            
        if it >= nMaxIt:
            print('MetodoDelleSecanti -- Numero massimo di iterazioni raggiunto')
        
        return it, xit
    
    
    """
    Applico il metodo iterativo di punto fisso
    INPUT:
        G : Funzione G derivata da F con operazioni
        x0 : Valore di innesco
        tol : Tolleranza
        nMaxIt : Numero massimo di iterazioni
    OUTPUT:
        it : Numero di iterate fatte
        xit : Le ordinate all'i-esima iterata
    """
    @staticmethod
    def MetodoIterativoDiPuntoFisso(G, x0, tolx, nMaxIt):     
        xit = []
        xit.append(x0)
        
        x1 = G(x0)
        d = x1 - x0
        xit.append(x1)
        it = 1
        
        while it < nMaxIt and  abs(d) >= tolx * abs(x1):
            x0 = x1
            x1 = G(x0)
            d = x1-x0
            it += 1
            xit.append(x1)
           
    
        if it >= nMaxIt:
            print('MetodoIterativoDiPuntoFisso -- Numero massimo di iterazioni raggiunto')
        
        return it, xit
    
    """
    Applico il metodo di Newton
    INPUT:
       F : Funzione considerata
       DF : Derivata prima della F
       x0 : Il valore di innesco
       nMaxIt : Numero massimo di iterazioni
       tolf : Tolleranza sulla funzione
       tolx : Tolleranza sul valore della x
       nMaxIt : Numero massimo di iterazioni
    OUTPUT:
       it : Numero di iterate fatte
       xit : Le ordinate all'i-esima iterata
    """
    @staticmethod
    def MetodoDiNewton(F, DF, x0, tolx ,tolf, nMaxIt):     
        xit = []
        fx0 = F(x0)
        dfx0 = DF(x0)
        EPS = np.spacing(1)
        
        if abs(dfx0) > EPS:
            d = fx0 / dfx0
            x1 = x0 - d
            fx1 = F(x1)
            xit.append(x1)
            it = 0           
        else:
            print('MetodoDiNewton -- Derivata nulla in x0')
            return [], 0
        it = 1
        while it < nMaxIt and abs(fx1) >= tolf and  abs(d) >= tolx * abs(x1):
            x0 = x1
            fx0 = F(x0)
            dfx0 = DF(x0)
            
            if abs(dfx0) > EPS:
                d = fx0 / dfx0
                x1 = x0 - d
                fx1 = F(x1)
                xit.append(x1)
                it += 1
            else:
                 print('MetodoDiNewton -- Derivata nulla in x0')
                 return it, xit           
           
        if it >= nMaxIt:
            print('Newton -- Raggiunto massimo numero di iterazioni')
        
        return it, xit
    
    """
    Applico il metodo di Newton Modificato
    INPUT :
       F : Funzione considerata
       DF : Derivata prima della F
       x0 : Il valore di innesco
       m : 
       tolf : Tolleranza sulla funzione
       tolx : Tolleranza sul valore della x
       nMaxIt : Numero massimo di iterazioni
    OUTPUT:
       it : Numero di iterate fatte
       xit : Le ordinate all'i-esima iterata
    """
    @staticmethod
    def MetodoDiNewtonModificato(F, DF, x0, m, tolx ,tolf, nMaxIt):     
        EPS = np.spacing(1)     
        xit = []
        fx0 = F(x0)
        dfx0 = DF(x0)
        
        if abs(dfx0) > EPS:
            d=fx0 / dfx0
            x1=x0 - m * d
            fx1 = F(x1)
            xit.append(x1)
            it = 0
           
        else:
            print('MetodoDiNewtonModificato -- Derivata nulla in x0')
            return [], 0
        
        it = 1
        while it < nMaxIt and abs(fx1)>=tolf and  abs(d) >= tolx * abs(x1):
            x0 = x1
            fx0 = F(x0)
            dfx0 = DF(x0)
            
            if abs(dfx0) > EPS:
                d = fx0/dfx0
                x1 = x0 - m * d
                fx1 = F(x1)
                xit.append(x1)
                it +=1
            else:
                print('MetodoDiNewtonModificato -- Derivata nulla in x0')
                return [], 0         
           
        if it == nMaxIt:
            print('MetodoDiNewtonModificato -- Raggiunto massimo numero di iterazioni')
        
        return it, xit

class SistemiLineari : 
    """  
    Risoluzione con procedura forward di Ax=b con A matrice triangolare inferiore  
    INPUT: 
        A : Matrice triangolare inferiore
        b : Termine noto
    OUTPUT: 
        x : soluzione del sistema lineare
        True, se sono soddisfatti i test di applicabilità
        False, se non sono soddisfatti
    """
    @staticmethod
    def __ForwardPropagation__(A, b) :
        # m = numero di righe
        # n = numero di colonne
        m, n = A.shape
        
        # Test dimensione
        if n != m:
            print('ForwardPropagation -- Errore [Matrice non quadrata]')
            return [], False
    
        # Test singolarita'
        if np.all(np.diag(A)) != True :
            print('ForwardPropagation -- Errore [elemento sulla diagonale nullo]')
            return [], False
    
        # Preallocazione vettore soluzione e azzeramento
        x = np.zeros((n,1))
    
        for i in range(n):
            # scalare = vettore riga * vettore colonna
            s = np.dot(A[i,:i], x[:i]) 
            x[i] = (b[i] - s) / A[i,i]
        
        return x, True
    
    """  
    Risoluzione con procedura backward propagation di Ax = b con A matrice triangolare superiore  
    INPUT: 
        A : Matrice triangolare superiore
        b : Termine noto
    OUTPUT: 
        x : soluzione del sistema lineare
        True, se sono soddisfatti i test di applicabilità
        False, se non sono soddisfatti
    """
    @staticmethod
    def BackwardPropagation(A, b) :
        # m = numero di righe
        # n = numero di colonne
        m,n = A.shape
        
        # Test dimensione
        if n != m:
            print('BackwardPropagation -- Errore [Matrice non quadrata]')
            return [], False
    
        # Test singolarita'
        if np.all(np.diag(A)) != True :
            print('BackwardPropagation -- Errore [elemento sulla diagonale nullo]')
            return [], False
    
        # Preallocazione vettore soluzione e azzeramento
        x = np.zeros((n,1))
    
        for i in range(n-1, -1, -1):
            # scalare = vettore riga * vettore colonna
            s = np.dot(A[i, i+1 : n], x[i+1 : n]) 
            x[i] = (b[i] - s) / A[i,i]
        
        return x, True
    
    """  
    Fattorizzazione P*A = L*U senza vettorizzazione
    INPUT:
        A : Matrice associata al sistema lineare
        Pivot : True -> usa il pivot, False -> non usa il pivot
    OUTPUT: 
        P : matrice identità
        L : matrice triangolare inferiore
        U : matrice triangolare superiore
        True, se sono soddisfatti i test di applicabilità
        False, se non sono soddisfatti
        t.c L*U = P*A = A
    """
    @staticmethod
    def __EliminazioneGaussiana__(A, Pivot):
        # Test dimensione
        m, n = A.shape
        if n != m:
            print("EliminazioneGaussiana -- Matrice Non Quadrata")
            return [], [], [], False  
        
        # Matrice identita' di dimensione n
        P = np.eye(n)
        # Copia di A per non sporcare l'input
        U = A.copy()
        
        # Fattorizzazione
        for k in range(n-1):
            
            if(Pivot) : 
                #Fissata la colonna k-esima calcolo l'indice di riga p a cui appartiene l'elemento di modulo massimo
                p = np.argmax(abs(U[k:n,k])) + k
                if p != k:
                  SistemiLineari.__SwapRows__(P,k,p)
                  SistemiLineari.__SwapRows__(U,k,p)
            elif U[k,k] == 0: # Test pivot se null e voglio la versione senza il pivot parziale
                print('EliminazioneGaussiana -- Elemento Diagonale Nullo')
                return P, [], [], [], False

            # Eliminazione gaussiana
            for i in range(k+1, n):
                U[i, k] = U[i, k] / U[k, k]
                # Memorizza i moltiplicatori nella parte di matrice che non uso
                for j in range(k+1, n): # Eliminato nella versione EliminazioneGaussianaSenzaPivotVettorizzata
                    # Eliminazione gaussiana sulla matrice
                    U[i, j] = U[i, j] - U[i, k] * U[k, j]
     
        # Estrae i moltiplicatori 
        L = np.tril(U,-1)+np.eye(n)
        # Estrae la parte triangolare superiore + diagonale
        U = np.triu(U)           
        return P, L, U, True
    
    """  
    Fattorizzazione P*A = L*U vettorizzata non ottimizzata
    INPUT:
        A : Matrice associata al sistema lineare
        Pivot : True -> usa il pivot, False -> non usa il pivot
    OUTPUT: 
        P : matrice identità
        L : matrice triangolare inferiore
        U : matrice triangolare superiore
        True, se sono soddisfatti i test di applicabilità
        False, se non sono soddisfatti
        t.c L*U = P*A = A
    """
    @staticmethod
    def __EliminazioneGaussianaVettorizzata1__(A, Pivot):
        # Test dimensione
        m, n = A.shape
        if n != m:
            print("EliminazioneGaussianaVettorizzata1 -- Matrice Non Quadrata")
            return [], [], [], False  
        
        # Matrice identita' di dimensione n
        P = np.eye(n)
        # Copia di A per non sporcare l'input
        U = A.copy()
        
        # Fattorizzazione
        for k in range(n-1):
            
            if(Pivot) : 
                #Fissata la colonna k-esima calcolo l'indice di riga p a cui appartiene l'elemento di modulo massimo
                p = np.argmax(abs(U[k:n,k])) + k
                if p != k:
                  SistemiLineari.__SwapRows__(P,k,p)
                  SistemiLineari.__SwapRows__(U,k,p)
            elif U[k,k] == 0: # Test pivot se null e voglio la versione senza il pivot parziale
                print('EliminazioneGaussianaVettorizzata1 -- Elemento Diagonale Nullo')
                return P, [], [], [], False

            # Eliminazione gaussiana
            for i in range(k+1, n):
                # Memorizza i moltiplicatori nella parte di matrice che non uso
                U[i, k] = U[i, k] / U[k, k]
                # Eliminazione gaussiana sulla matrice
                U[i, k+1 : n] = U[i, k+1 : n] - U[i, k] * U[k, k+1 : n]
     
        # Estrae i moltiplicatori 
        L = np.tril(U,-1)+np.eye(n)
        # Estrae la parte triangolare superiore + diagonale
        U = np.triu(U)           
        return P, L, U, True
    
    """  
    Fattorizzazione P*A = L*U vettorizzata ottimizzata
    INPUT:
        A : Matrice associata al sistema lineare
        Pivot : True -> usa il pivot, False -> non usa il pivot
    OUTPUT: 
        P : matrice identità
        L : matrice triangolare inferiore
        U : matrice triangolare superiore
        True, se sono soddisfatti i test di applicabilità
        False, se non sono soddisfatti
        t.c L*U = P*A = A
    """
    @staticmethod
    def __EliminazioneGaussianaVettorizzata2__(A, Pivot):
        # Test dimensione
        m, n = A.shape
        if n != m:
            print("EliminazioneGaussianaVettorizzata2 -- Matrice Non Quadrata")
            return [], [], [], False  
        
        # Matrice identita' di dimensione n
        P = np.eye(n)
        # Copia di A per non sporcare l'input
        U = A.copy()
        
        # Fattorizzazione
        for k in range(n-1):
            
            if(Pivot) : 
                #Fissata la colonna k-esima calcolo l'indice di riga p a cui appartiene l'elemento di modulo massimo
                p = np.argmax(abs(U[k:n,k])) + k
                if p != k:
                  SistemiLineari.__SwapRows__(P,k,p)
                  SistemiLineari.__SwapRows__(U,k,p)
            elif U[k,k] == 0: # Test pivot se null e voglio la versione senza il pivot parziale
                print('EliminazioneGaussianaVettorizzata2 -- Elemento Diagonale Nullo')
                return P, [], [], False

            # Eliminazione gaussiana
            # Memorizza i moltiplicatori
            U[k+1 : n, k] = U[k+1 : n, k] / U[k, k]
            # Eliminazione gaussiana sulla matrice
            U[k+1 : n, k+1 : n] = U[k+1 : n, k+1 : n] - np.outer(U[k+1 : n, k], U[k, k+1 : n])  
     
        # Estrae i moltiplicatori 
        L = np.tril(U,-1)+np.eye(n)
        # Estrae la parte triangolare superiore + diagonale
        U = np.triu(U)           
        return P, L, U, True
    
    """  
    Scambia la k-esima riga con la p-esima riga
    INPUT:
        A : Matrice associata al sistema lineare
        k : Indice k-esimo
        p : Indice p-eseimo
    OUTPUT:
        A con le righe scambiate
    """
    @staticmethod
    def __SwapRows__(A, k, p):
        A[[k,p], :] = A[[p,k], :]
        
    """
    Risolve un qualsiasi sistema lineare Ax=b
    INPUT: 
        A : Matrice associata al sistema lineare
        b : Vettore dei termini noti
    OUTPUT:
        x : La soluzione del sistema
        True, se sono soddisfatti i test di applicabilità
        False, se non sono soddisfatti
    """
    @staticmethod
    def Solve(A, b, Pivot = False):
        # Test dimensione
        m, n = A.shape
        if n != m:
            print("Solve -- Matrice Non Quadrata")
            return [], [], [], False
        
        Pivot = False
        
        for i in range(1, n):
            if(np.linalg.det(A[:i,:i]) == 0):
                print("Solve -- Det(A" + str(i ) + "" + str(i) + ") Nullo -- Bisogna usare per forza il pivoting")
                Pivot = True
                break
        
        P, L, U, cond = SistemiLineari.__EliminazioneGaussianaVettorizzata2__(A, Pivot)
        Pb = np.dot(P, b)
        y, cond = SistemiLineari.__ForwardPropagation__(L, Pb)
        
        if cond == True:
            x, cond = SistemiLineari.__BackwardPropagation__(U, y)
        else:
            return [], False

        return x, True
    
class ApprossimazioneMinimiQuadrati :
    
    """
    Applico il metodo QRLS per trovare il polinomio di grado n che approssima al meglio i dati 
    INPUT:
        x : Vettore colonna con le ascisse dei punti
        y : Vettore colonna con le ordinate dei punti 
        n : Grado del polinomio approssimante
    OUTPUT:
        sol : Vettore colonna contenente i coefficienti incogniti
    """
    @staticmethod
    def MetodoQRLS(x, y, n):
        
        H = np.vander(x, n+1)
        
        Q, R = spl.qr(H)
        
        y1 = np.dot(Q.T, y)
        
        sol, cond = SistemiLineari.BackwardPropagation(R[0 : n+1, :], y1[0 : n+1])
        
        return  sol

class InterpolazionePolinomiale :
    
    """
    Questo metodo restituisce il k-esimo elemento della Base di Lagrange
    INPUT:
        xnodi : Tutti i nodi
        k : Indice considerato
    OUTPUT:
        k-esimo elemento della Base di Lagrange
    """
    @staticmethod
    def __BaseDiLagrange__(xnodi, k):
        xzeri = np.zeros_like(xnodi)
        n = xnodi.size
     
        if k == 0:
            xzeri = xnodi[1:n]
        else:
            xzeri = np.append(xnodi[0:k],xnodi[k+1:n])
    
        num = np.poly(xzeri) 
        den = np.polyval(num,xnodi[k])
        p = num / den
    
        return p
    
    """"
    Funzione che determina in un insieme di punti il valore del polinomio
    Interpolante ottenuto dalla formula di Lagrange.
    INPUT:
        x : Vettore con i nodi dell'interpolazione
        f : Vettore con i valori dei nodi 
        xx : Vettore con i punti in cui si vuole calcolare il polinomio
    OUTPUT:
        y : Vettore contenente i valori assunti dal polinomio interpolante
    """
    @staticmethod
    def Interpola(x, f, xx):
        n = x.size
        m = xx.size
        L = np.zeros( (n,m) )
        
        for k in range(n):
            p = InterpolazionePolinomiale.__BaseDiLagrange__(x, k)
            L[k, :] = np.polyval(p, xx)
        
        return np.dot(f, L)
 
class IntegrazioneNumerica :
    
    """
    Questa formula restituisce il valore dell'integrale tra [a,b] approsimata con la formula del trapezio
    suddividendo [a,b] in n sotto-intervalli
    INPUT:
        f: Funzione considerata
        a: Estremo inferiore
        b: Estremo superiore
        n: Numero di sotto-intervalli
    OUTPUT:
        L'area calcolata
    """
    @staticmethod
    def __TrapezioComposita__(f, a, b, n):        
        h = (b - a) / n
        nodi = np.arange(a, b + h, h)
        fn = f(nodi)
        I = (fn[0] + 2*np.sum(fn[1 : n]) + fn[n]) * h / 2
        return I
    
    """
    Questa formula restituisce il valore dell'integrale tra [a,b] approsimata con la formula di Simpson 
    suddividendo [a,b] in n sotto-intervalli
    INPUT:
        f: Funzione considerata
        a: Estremo inferiore
        b: Estremo superiore
        n: Numero di sotto-intervalli
    OUTPUT:
        L'area calcolata
    """
    @staticmethod
    def __SimpsonCompita__(f, a, b, n):        
        h = (b - a) / (2 * n)
        nodi = np.arange(a, b + h, h)
        fn = f(nodi)
        I = (fn[0] + 2 * np.sum(fn[2 : 2*n : 2]) + 4 * np.sum(fn[1 : 2*n : 2]) + fn[2*n]) * h / 3
        return I
    
    """
    Questa formula restituisce il valore dell'integrale tra [a,b] approsimata con la formula di Simpson
    INPUT:
        f: Funzione considerata
        a: Estremo inferiore
        b: Estremo superiore
        tol: Tolleranza dell'errore
    OUTPUT:
        L'area calcolata
    """
    @staticmethod
    def TrapezioCompositaWithTollerance(f, a, b, tol):
        Nmax = 2048
        err = 1        
        N = 1
        IN = IntegrazioneNumerica.__TrapezioComposita__(f, a, b, N)
        
        while N <= Nmax and err > tol :
            N = 2 * N
            I2N = IntegrazioneNumerica.__TrapezioComposita__(f, a, b, N)
            err = abs(IN - I2N) / 3
            IN = I2N
        
        if N > Nmax:
            print('TrapezioCompositaWithTollerance -- Raggiunto numero massimo di intervalli')
            
        return IN, N
    
    """
    Questa formula restituisce il valore dell'integrale tra [a,b] approsimata con la formula di Simpson
    INPUT:
        f: Funzione considerata
        a: Estremo inferiore
        b: Estremo superiore
        tol: Tolleranza dell'errore
    OUTPUT:
        L'area calcolata
    """
    @staticmethod
    def SimpsonCompositaWithTollerance(f, a, b, tol):
        Nmax = 2048
        err = 1        
        N = 1
        IN = IntegrazioneNumerica.__SimpsonCompita__(f, a, b, N)
        
        while N <= Nmax and err > tol :
            N = 2 * N
            I2N = IntegrazioneNumerica.__SimpsonCompita__(f, a, b, N)
            err = abs(IN - I2N) / 15
            IN = I2N
        
        if N > Nmax:
            print('SimpsonCompositaWithTollerance -- Raggiunto numero massimo di intervalli')
            
        return IN, N
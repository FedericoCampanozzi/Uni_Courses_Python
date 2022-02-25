def somma (x, y):
    return x + y

lista = [1, 2, 3, 4]
lsommata = [somma(lista[i],lista[i+1]) for i in range(0, len(lista), 2) ]
lsommata2 = [lista[i] + lista[i+1] for i in range(0, len(lista), 2) ]

print(lista)
print(lsommata)
print(lsommata2)
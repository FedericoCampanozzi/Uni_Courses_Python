import sys
sys.path.append('../RemoteData')

import RemoteData
import numpy as np
import csv

RemoteData.RetriveDataFrom("https://git.io/fhxQh", "purchases_data.zip") # Scarico i dati che mi servono per l'elaborazione

with open("data/users.csv", "r") as f:
    users = {int(uid): name for uid, name in csv.reader(f, delimiter=";")} # Mappa id utente => nome utente
with open("data/items.csv", "r") as f:
    items = {int(iid): name for iid, name in csv.reader(f, delimiter=";")} # Mappa id prodotto => nome prodotto

N = 20 # Numero di suggerimenti

# Creo una mappa dove memorizzo sulle chiavi l'id utente e come valori
# l'indice nell'array users
# enumerate ritorna per ogni valore 0,1,2,..., len(users) l'indice dell'utente nell'array users
user_indices = {uid: index for index, uid in enumerate(sorted(users.keys()))}
# Similmente creo la mappa per i prodotti
item_indices = {iid: index for index, iid in enumerate(sorted(items.keys()))}

print(f"{len(users)} utenti, {len(items)} prodotti")

# purchases è una matrice binaria che memorizza 1 se l'i-esimo utente ha comprato il j-esimo prodotto
# 0 altrimenti ===> np.zeros(shape della matrice utenti x prodotti) 
purchases = np.zeros((len(users), len(items)), dtype=int)

with open("data/purchases-2000.csv", "r") as f:
    for uid, iid in csv.reader(f, delimiter=";"):
        purchases[user_indices[int(uid)], item_indices[int(iid)]] = 1

# prodotti classico riga-colonna con operatore @ 
similarity = purchases @ purchases.T 
interest = similarity @ purchases  
interest[purchases.astype(bool)] = 0 # modelliamo assumendo che per i prodotti già comprati non c'è interesse
interest_ranking = (-interest).argsort(1).argsort(1) # per ogni riga abbiamo ordinato l'indici dei prodotti con più interesse
suggestions = (interest_ranking < N).astype(int) # applichiamo un filtro

# zeros_like => crea una matrice della stessa shape della m. passata come argomento
purchases_updated = np.zeros_like(purchases) # oppure np.zeros(purchases.shape)

with open("data/purchases-2014.csv", "r") as f:
    for uid, iid in csv.reader(f, delimiter=";"):
        purchases_updated[user_indices[int(uid)], item_indices[int(iid)]] = 1


new_purchases = purchases_updated - purchases # sono i prodotti comprati tra il 2000 e il 2014
hits = suggestions * new_purchases # sono i prodotti suggeriti che poi sono stati comprati 
satisfied_users = hits.max(1) # sono gli soddisfatti => quelli per cui almeno un suggerimento è stato utile
satPerc = satisfied_users.mean() * 100 # % degli utenti soddisfatti 
print(f"Soddisfazione modello base : {satPerc:5.2f} %")

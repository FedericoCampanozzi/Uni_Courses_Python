import sys
sys.path.append('../RemoteData')

import RemoteData
import numpy as np
import csv

RemoteData.RetriveDataFrom("https://git.io/fhxQh", "purchases_data.zip")

with open("data/users.csv", "r") as f:
    users = {int(uid): name for uid, name in csv.reader(f, delimiter=";")}
with open("data/items.csv", "r") as f:
    items = {int(iid): name for iid, name in csv.reader(f, delimiter=";")}
with open("data/purchases-2000.csv", "r") as f:
    purchases_set = {(int(uid), int(iid)) for uid, iid in csv.reader(f, delimiter=";")}
    
user_indices = {uid: index for index, uid in enumerate(sorted(users.keys()))}
item_indices = {iid: index for index, iid in enumerate(sorted(items.keys()))}

n_users = len(users)
n_items = len(items)

print(f"{n_users} utenti, {n_items} prodotti")

purchases = np.zeros((n_users, n_items), dtype=int)

with open("data/purchases-2014.csv", "r") as f:
    for uid, iid in csv.reader(f, delimiter=";"):
        purchases[user_indices[int(uid)], item_indices[int(iid)]] = 1

similarity = purchases @ purchases.T        
interest = similarity @ purchases
interest[purchases.astype(bool)] = 0

#new_purchases = purchases_updated - purchases
#hits = suggestions * new_purchases
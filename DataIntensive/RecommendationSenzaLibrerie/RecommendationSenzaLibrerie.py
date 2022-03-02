#import sys
#sys.path.append('../RemoteData')

import RemoteData
RemoteData.RetriveDataFrom("https://git.io/fhxQh", "purchases_data.zip")

# Modello con formulazione base di un modello di similarita'
import csv
N = 20

# Reading data ==> csv file
with open("data/purchases-2000.csv", "r") as f:
    purchases = set(
        (int(uid), int(iid))
        for uid, iid
        in csv.reader(f, delimiter=";")
    )

with open("data/users.csv", "r") as f:
    users = {
        int(uid): name
        for uid, name
        in csv.reader(f, delimiter=";")
    }

with open("data/items.csv", "r") as f:
    items = {int(iid): name for iid, name in csv.reader(f, delimiter=";")}

purchases_by_user = {}
for uid, iid in purchases:
    purchases_by_user.setdefault(uid, set()).add(iid)

purchases_by_item = {}
for uid, iid in purchases:
    purchases_by_item.setdefault(iid, set()).add(uid)
    
def user_similarity(uid1, uid2):
    """Count products purchased by both given users."""
    return len(purchases_by_user[uid1] & purchases_by_user[uid2])

# dati precalcolati per risparmiare tempo
# invece di invocare la funzione user_similarity() leggiamo i valori sulla mappa user_similarities
user_similarities = {
    (i, j): user_similarity(i, j)
    for i in users.keys()
    for j in users.keys()
    if i != j
}

def interest(uid, iid):
    """Estimate the interest of given user for given product."""
    return sum(
        user_similarities[(uid, ouid)]
        for ouid in purchases_by_item[iid]
        if uid != ouid
    )

interests_by_user = {
    uid: {
        iid: interest(uid, iid)
        for iid in items.keys()
        if iid not in purchases_by_user[uid]
    } for uid in users.keys()
}


def suggest(uid):
    """Recommend N products to given user."""
    interests = interests_by_user[uid].items()
    sorted_interests = sorted(interests, key=lambda x: x[1], reverse=True)
    return set(iid for iid, score in sorted_interests[:N])

suggestions_by_user = {uid: suggest(uid) for uid in users.keys()}

with open("data/purchases-2014.csv", "r") as f:
    purchases_updated = set(
        (int(uid), int(iid))
        for uid, iid
        in csv.reader(f, delimiter=";")
    )
    
new_purchases = purchases_updated - purchases

new_purchases_by_user = {}
for uid, iid in new_purchases:
    new_purchases_by_user.setdefault(uid, set()).add(iid)
    
satisfied_users = {
    uid for uid in users.keys()
    if suggestions_by_user[uid] & new_purchases_by_user.get(uid, set())
}

satPerc = round(len(satisfied_users) / len(users) * 100, 3)
print(f"Soddisfazione modello base : {satPerc:5.2f} %")

# Modello Casuale
import random

def suggest_random(uid):
    return {random.randint(0, len(users)) for i in range(20)}

results = []
for i in range(1000):
    random.seed(i)
    suggestions_by_user_random = {uid: suggest_random(uid) for uid in users.keys()}
    randomly_satisfied_users = {
        uid for uid in users.keys()
        if suggestions_by_user_random[uid] & new_purchases_by_user.get(uid, set())
    }
    satRandPerc = round(len(randomly_satisfied_users) / len(users) * 100, 3)
    results.append(satRandPerc);
    
satRandPercAvg = sum(results) / len(results)
print(f"Soddisfazione media del modello casuale {len(results)} : {satRandPercAvg:5.2f} %")
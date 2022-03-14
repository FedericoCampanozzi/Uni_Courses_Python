import os.path
import numpy as np
import pandas as pd

if not os.path.exists("tips.csv"):
    from urllib.request import urlretrieve
    urlretrieve("https://git.io/JYJNP", "tips.csv")

tips = pd.read_csv("tips.csv")

custom_dtypes = {
    "sex": "category",
    "smoker": "category",
    "day": "category",
    "time": "category",
}

tips = pd.read_csv("tips.csv", dtype=custom_dtypes)

print(f"\n1a - numero di persone nella prima riga della tabella (la cui etichetta è 0) : {tips.loc[0, 'size']}")
print(f"\n1b -  tutti i dati (cioè l'intera riga) relativi al tavolo col totale del conto più alto : \n{tips.loc[tips['total_bill'].idxmax()]}")
print(f"\n1c - media delle mance ricevute : {tips['tip'].mean()}")
print(f"\n1d - spesa media per persona più alta in un tavolo : {np.max(tips['total_bill'] / tips['size'])}")


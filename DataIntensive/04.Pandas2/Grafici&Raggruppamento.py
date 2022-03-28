import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print(f"\n\n2a - Estrarre le statistiche di base (describe) sulla spesa media per persona ai tavoli : \n{(tips['total_bill'] / tips['size']).describe()}")
print(f"\n2b - Suddividere la percentuale della mancia rispetto al totale in 3 fasce di uguale ampiezza ed estrarre il numero di conti in ciascuna : \n{pd.cut(tips['tip'] / tips['total_bill'], 3).value_counts()}")

# Creare i seguenti grafici:
# 3a - un grafico a torta con la distribuzione dei totali dei conti in tre fasce di uguale ampiezza
pd.cut(tips.loc[:,"total_bill"],3).value_counts().plot.pie()
plt.show()

# 3b - un grafico a torta con la distribuzione delle sole cene (time="Dinner") per giorno della settimana
tips[tips["time"] == "Dinner"]["day"].value_counts().plot.pie()
plt.show()

# 3c - un'istogramma con la distribuzione del rapporto tra mancia e totale del conto
plt.hist(tips["tip"] / tips["total_bill"])
plt.show()

# 3d - un box plot con la stessa distribuzione (3c)
(tips["tip"] / tips["total_bill"]).plot.box()
plt.show()

# 3e - un grafico a dispersione con la correlazione tra mancia lasciata e numero di persone al tavolo
tips.plot.scatter("tip", "size")
plt.show()

# 3f - una figura con un grafico a barre con la distribuzione di tavoli fumatori e non per ciascun giorno della settimana
for n, day in enumerate(["Thur", "Fri", "Sat", "Sun"], start=1):
    tips.loc[tips["day"] == day, "smoker"].value_counts().plot.bar(ax=plt.subplot(2, 2, n), title=day)
plt.show()

print(f"\n\n4a - Ottenere una serie col numero di persone medio a tavoli con fumatori e non :\n{tips.groupby('smoker')['size'].mean()}")
print(f"\n4b - Ottenere un frame col totale e la media delle mance per ciascun giorno della settimana :\n{tips.groupby('day')['tip'].agg(['mean', 'sum'])}")
print("\n4c - Ottenere una serie con la media delle mance ricevute di venerdì ('Fri'), suddivise tra pranzo e cena :\n" +
      f"{tips[tips['day'] == 'Fri'].groupby('time')['tip'].mean()}")
print(f"\n4d - Ottenere una serie con la media delle mance ricevute tra tavoli che hanno pagato almeno 20$ e quelli che hanno pagato di meno :\n{tips.groupby(tips['total_bill'] >= 20)['tip'].mean()}")

print("\n\n5a - Ottenere tramite groupby un frame col numero di conti per ogni giorno della settimana lungo le righe e ogni pasto lungo le colonne :\n" + 
      f"{tips.groupby(['day', 'time']).size().unstack('time')}")
print("\n5b - Ottenere tramite pivot_table un frame col totale di mance ottenute per ogni giorno della settimana lungo le righe e ogni pasto lungo le colonne :\n" +
      f"{tips.pivot_table(values='tip',index='day',columns='time',aggfunc='sum')}")

# 5c - Dividere i totali dei conti in tre fasce di uguale ampiezza ed ottenere un frame con una riga per ciascuna fascia che riporti la media di totali e mance divise per pranzi e cene
tips.pivot_table(
    values=["total_bill","tip"],
    index=pd.cut(tips["total_bill"],3),
    columns="time",
    aggfunc=["sum","mean"],
).plot.bar();
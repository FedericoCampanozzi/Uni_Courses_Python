import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt

if not os.path.exists("usa_census.npz"):
    from urllib.request import urlretrieve
    urlretrieve("https://git.io/vxh8Y", "usa_census.npz")
    
data = np.load("usa_census.npz", allow_pickle=True)
states = data["states"]
population = data["population"]
area = data["area"]


print(f"\n1a - il numero di abitanti del 5° stato in ordine alfabetico (California) : {population[4]}")
print(f"1b - i nomi degli ultimi tre stati in ordine alfabetico : {states[-3:]}")
print(f"1c - il numero di abitanti in Florida (senza sapere a priori la sua posizione) : {population[states == 'Florida'][0]}")
print(f"1d - i nomi degli stati con almeno 20 milioni di abitanti : {states[population >= 2e7]}")
print(f"1e - il numero totale di abitanti in tutti gli stati : {population.sum()}")
print(f"1f - il nome dello stato con meno abitanti : {states[population == population.min()][0]}")
print(f"1g - il nome dello stato con più abitanti : {states[population == population.max()][0]}")

population = pd.Series(data["population"], index=data["states"])

#2a - La serie area riporta la superficie degli stati in miglia quadrate: ricavare una serie area_km2 con la superficie in chilometri quadrati (1 mi² = 2,59 km²)
area_km2 = area * 2.59
#2b - Creare una serie density con la densità di popolazione di ciascuno stato in abitanti per km²
density = population / area_km2

west_coast = ["Washington", "Oregon", "California"]

area        = pd.Series(data["area"],        index=data["states"])
other_state = pd.Series(data["other_state"], index=data["states"])
from_abroad = pd.Series(data["from_abroad"], index=data["states"])

print(f"\n3a - la densità di popolazione dello stato più piccolo : {density[area.idxmin()]}")
print(f"3b - il numero di stati la cui popolazione è superiore al milione di abitanti : {len(states[population >= 1e6])}")
print(f"3c - il totale della popolazione degli stati sulla costa ovest (usare lista west_coast definita sopra) : {population[west_coast].sum()}")
print(f"3d - la densità media degli stati con almeno 10 milioni di abitanti : {density[population > 1e7].mean()}")

census = pd.DataFrame({
    "population": population,
    "from_abroad": from_abroad,
    "area": area_km2
})

state_to_state = pd.DataFrame(data["state_to_state"], index=data["states"], columns=data["states"])

print(f"\n4a - la superficie dello stato più grande : {census['area'].idxmax()}")
print(f"4b -il numero totale di persone emigrate dall'Arizona ad un altro stato : {state_to_state['Arizona'].sum()}")
print(f"4c - la densità media degli stati con almeno 10 milioni di abitanti : {state_to_state.sum(axis=1).idxmin()}")


print(f"\n5a - la superficie della California : {census['area']['California']}")
print(f"5b - la popolazione (colonna 0) del 13° stato nella tabella : {census.iloc[12, 0]}")
print(f"5c - la popolazione totale degli stati con nome che inizia per M : {census.loc['N': 'M', 'population']}")
print(f"5d - la superficie complessiva degli stati con almeno 20 milioni di abitanti : {census.loc[census['population'] > 2e7, 'area'].sum()}")
print(f"5e - la popolazione media degli stati con almeno l'1% di popolazione immigrato dall'estero (from_abroad) nell'ultimo anno : {state_to_state.sum(axis=1).idxmin()}")
print(f"5f - la superficie totale dei 5 stati con densità di popolazione minore : {state_to_state.sum(axis=1).idxmin()}")
print(f"5g - la popolazione (colonna 0) del 3° stato con superficie maggiore : {state_to_state.sum(axis=1).idxmin()}")


population_mln = population / 1e6
plt.figure(figsize=(20, 4))
population_mln.plot.bar();

plt.figure(figsize=(4, 6))
plt.boxplot(population_mln, showmeans=True)
plt.grid(axis="y");

population_mln.describe()

np.random.seed(123)
random_values = np.random.normal(size=1000)
plt.figure(figsize=(4, 6))
plt.boxplot(random_values, showmeans=True)
plt.grid(axis="y")

census.plot.box(showmeans=True)
plt.grid(axis="y")

# grafico solo su due colonne, parzialmente più leggibile
census[["from_abroad", "area"]].plot.box(showmeans=True)
plt.grid(axis="y")
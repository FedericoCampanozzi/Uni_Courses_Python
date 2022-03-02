import os
from urllib.request import urlretrieve
from zipfile import ZipFile

def RetriveDataFrom(remotePath, fileName):
    # se il file "purchases_data.zip" non esiste
    if not os.path.exists("data"):
        # scarica il file dall'URL indicato
        urlretrieve(remotePath, fileName)
        # apri il file zip ed estrai tutto il contenuto nella directory corrente
        with ZipFile("purchases_data.zip") as f:
            f.extractall("data")
        # muovo purchases_data.zip nella cartella data
        os.replace(fileName, "data/" + fileName)
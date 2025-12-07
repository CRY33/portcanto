# Clustering de Ciclistes - Port del Cantó

Projecte d'anàlisi de clustering per identificar diferents perfils de ciclistes segons els seus temps de pujada i baixada al Port del Cantó.

## Descripció

Aquest projecte utilitza l'algorisme **K-means** per classificar ciclistes en 4 perfils segons el seu rendiment:

- **BEBB**: Bons Escaladors, Bons Baixadors
- **BEMB**: Bons Escaladors, Mals Baixadors  
- **MEBB**: Mals Escaladors, Bons Baixadors
- **MEMB**: Mals Escaladors, Mals Baixadors

## Instal·lació
```bash
# Crear entorn virtual
python -m venv venv

# Activar entorn virtual
.\venv\Scripts\Activate.ps1

# Instal·lar dependències
pip install -r requirements.txt
```

## Ús

### 1. Generar dataset
```bash
python generardataset.py
```

Genera un fitxer `data/ciclistes.csv` amb 100 ciclistes sintètics.

### 2. Executar clustering
```bash
python clustersciclistes.py
```

Realitza el clustering, genera visualitzacions i guarda el model.

### 3. Experiments amb MLflow
```bash
python mlflowtracking-K.py
mlflow ui
```

Prova diferents valors de K (2-8) i visualitza els resultats a `http://localhost:5000`

### 4. Executar tests
```bash
python -m unittest tests.testportcanto -v
```

## Estructura del projecte

portcanto/
├── data/              # Datasets i visualitzacions
├── model/             # Models guardats
├── img/               # Gràfiques dels clusters
├── informes/          # Fitxers de ciclistes per tipus
├── tests/             # Tests unitaris
└── mlruns/            # Experiments MLflow

## Tecnologies

- Python 3.13
- scikit-learn (K-means clustering)
- pandas, numpy (processament de dades)
- matplotlib, seaborn (visualitzacions)
- MLflow (tracking d'experiments)

## Autor

Toni Carballo - EAC6 - IOC


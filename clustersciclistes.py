"""
@ IOC - CE IABD TONI CARBALLO EAC6
"""
import os
import logging
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score


def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes

    arguments:
            path -- dataset

    Returns: dataframe
    """

    return pd.read_csv(path)


def eda(df):
    """
    Exploratory Data Analysis del dataframe

    arguments:
            df -- dataframe

    Returns: None
    """
    print("=" * 60)
    print("ANÀLISI EXPLORATORI DE DADES (EDA)")
    print("=" * 60)

    # Informació general del dataset
    print("\n1. INFORMACIÓ GENERAL DEL DATASET")
    print("-" * 60)
    print(f"Nombre de files: {df.shape[0]}")
    print(f"Nombre de columnes: {df.shape[1]}")
    print(f"\nNoms de les columnes: {list(df.columns)}")

    # Tipus de dades
    print("\n2. TIPUS DE DADES")
    print("-" * 60)
    print(df.dtypes)

    # Primeres i últimes files
    print("\n3. PRIMERES 5 FILES")
    print("-" * 60)
    print(df.head())

    print("\n4. ÚLTIMES 5 FILES")
    print("-" * 60)
    print(df.tail())

    # Estadístiques descriptives
    print("\n5. ESTADÍSTIQUES DESCRIPTIVES")
    print("-" * 60)
    print(df.describe())

    # Valors nuls
    print("\n6. VALORS NULS")
    print("-" * 60)
    print(df.isnull().sum())

    # Distribució per tipus de ciclista
    print("\n7. DISTRIBUCIÓ PER TIPUS DE CICLISTA")
    print("-" * 60)
    print(df['tipus'].value_counts())

    # Estadístiques per tipus de ciclista
    print("\n8. ESTADÍSTIQUES PER TIPUS DE CICLISTA")
    print("-" * 60)
    print(df.groupby('tipus')[['temps_pujada', 'temps_baixada']].describe())

    print("\n" + "=" * 60)
    print("FI DE L'ANÀLISI EXPLORATORI")
    print("=" * 60 + "\n")

def clean(df):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

    arguments:
            df -- dataframe

    Returns: dataframe
    """

    # eliminem les columnes que no interessen
    df = df.drop('dorsal', axis=1)
    df = df.drop('tipus', axis=1)
    return df


def extract_true_labels(df):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)

    arguments:
            df -- dataframe

    Returns: numpy ndarray (true labels)
    """
    # Extreure la columna 'tipus' com a numpy array
    labels = df['tipus'].values

    return labels


def visualitzar_pairplot(df):
    """
    Genera una imatge combinant entre sí tots els parells d'atributs.
    Serveix per apreciar si es podran trobar clústers.

    arguments:
            df -- dataframe

    Returns: None
    """
    if 'tipus' in df.columns:
        sns.pairplot(df, hue='tipus', palette='Set2')
    else:
        sns.pairplot(df)

    plt.savefig('data/pairplot_ciclistes.png')
    plt.show()


def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
    Entrena el model

    arguments:
            data -- les dades: tp i tb

    Returns: model (objecte KMeans)
    """
    # Crear i entrenar el model KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)

    return model


def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

    arguments:
            data -- el dataset sobre el qual hem entrenat
            labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)

    Returns: None
    """
    try:
        os.makedirs(os.path.dirname('img/'))
    except FileExistsError:
        pass

        # Crear una còpia del dataframe i afegir la columna 'cluster'
    df_plot = data.copy()
    df_plot['cluster'] = labels
    # Gràfica 1: temps_pujada vs temps_baixada
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='temps_pujada',
        y='temps_baixada',
        data=df_plot,
        hue='cluster',
        palette="rainbow")
    plt.title('Clusters: Temps Pujada vs Temps Baixada')
    plt.savefig("img/grafica1.png", dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close()

    # Gràfica 2: distribució temps_pujada
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=df_plot.index,
        y='temps_pujada',
        data=df_plot,
        hue='cluster',
        palette="rainbow")
    plt.title('Clusters: Distribució Temps Pujada')
    plt.savefig("img/grafica2.png", dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close()

    # Gràfica 3: distribució temps_baixada
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=df_plot.index,
        y='temps_baixada',
        data=df_plot,
        hue='cluster',
        palette="rainbow")
    plt.title('Clusters: Distribució Temps Baixada')
    plt.savefig("img/grafica3.png", dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close()


def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

    arguments:
    tipus -- un array de tipus de patrons que volem actualitzar associant els labels
    model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    # proposta de solució

    dicc = {'tp': 0, 'tb': 1}

    logging.info('Centres:')
    for j in range(len(tipus)):
        logging.info('{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(j, model.cluster_centers_[
                     j][dicc['tp']], model.cluster_centers_[j][dicc['tb']]))

    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(clustering_model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if clustering_model.cluster_centers_[
            lst[0]][0] < clustering_model.cluster_centers_[
            lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus)
    return tipus


def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
    4 fitxers de ciclistes per cadascun dels clústers

    arguments:
            df -- dataframe
            tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """

    ciclistes_label = [
        df[df['label'] == 0],
        df[df['label'] == 1],
        df[df['label'] == 2],
        df[df['label'] == 3]
    ]

    try:
        os.makedirs(os.path.dirname('informes/'))
    except FileExistsError:
        pass

    for tip in tipus:
        fitxer = tip['name'] + '.txt'
        foutput = open("informes/" + fitxer, "w")

        t = [t for t in tipus if t['name'] == tip['name']]
        dorsals = ciclistes_label[t[0]['label']]['dorsal'].values

        # Escriure dorsals al fitxer
        for dorsal in dorsals:
            foutput.write(str(dorsal) + '\n')

        foutput.close()

    logging.info('S\'han generat els informes en la carpeta informes/\n')


def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

    arguments:
            dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
            model -- clustering model
    Returns: (dades agrupades, prediccions del model)
    """
    # Crear dataframe
    df_nous = pd.DataFrame(
        data=dades,
        columns=[
            'dorsal',
            'temps_pujada',
            'temps_baixada',
            'temps_total'])

    logging.info('\nNous ciclistes:\n%s', df_nous)

    # Predicció amb només temps_pujada i temps_baixada
    prediccions = model.predict(
        df_nous[['temps_pujada', 'temps_baixada']])
    return df_nous, prediccions

# ----------------------------------------------


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    """    
    load_dataset
    EDA
    clean
    extract_true_labels
    eliminem el tipus, ja no interessa .drop('tipus', axis=1)
    visualitzar_pairplot
    clustering_kmeans
    pickle.dump(...) guardar el model
    mostrar scores i guardar scores
    visualitzar_clusters
    """
    PATH_DATASET = './data/ciclistes.csv'
    ciclistes_data = load_dataset(PATH_DATASET)
    eda(ciclistes_data)
    true_labels = extract_true_labels(ciclistes_data)
    visualitzar_pairplot(ciclistes_data)
    ciclistes_clean = clean(ciclistes_data)
    # ciclistes_clean = ciclistes_clean.drop('tipus', axis=1) # eliminem el
    # tipus, ja no interessa

    clustering_model = clustering_kmeans(ciclistes_clean, n_clusters=4)
    # guardem el model
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)
    data_labels = clustering_model.labels_

    logging.info(
        '\nHomogeneity: %.3f',
        homogeneity_score(
            true_labels,
            data_labels))
    logging.info(
        'Completeness: %.3f',
        completeness_score(
            true_labels,
            data_labels))
    logging.info('V-measure: %.3f', v_measure_score(true_labels, data_labels))
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump({
            "h": homogeneity_score(true_labels, data_labels),
            "c": completeness_score(true_labels, data_labels),
            "v": v_measure_score(true_labels, data_labels)
        }, f)

    visualitzar_clusters(
        ciclistes_data[['temps_pujada', 'temps_baixada']], data_labels)

    # array de diccionaris que assignarà els tipus als labels
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'},
             {'name': 'MEBB'}, {'name': 'MEMB'}]

    """
	afegim la columna label al dataframe
	associar_clusters_patrons(tipus, clustering_model)
	guardem la variable tipus a model/tipus_dict.pkl
	generar_informes
	"""
    ciclistes_data['label'] = clustering_model.labels_.tolist()
    tipus = associar_clusters_patrons(tipus, clustering_model)
    # guardem la variable tipus
    with open('model/tipus_dict.pkl', 'wb') as f:
        pickle.dump(tipus, f)
    logging.info('\nTipus i labels:\n%s', tipus)

    # Classificació de nous valors
    nous_ciclistes = [
        [500, 3230, 1430, 4670],  # BEBB
        [501, 3300, 2120, 5420],  # BEMB
        [502, 4010, 1510, 5520],  # MEBB
        [503, 4350, 2200, 6550]  # MEMB
    ]

    """
	nova_prediccio

	#Assignació dels nous valors als tipus
	for i, p in enumerate(pred):
		t = [t for t in tipus if t['label'] == p]
		logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
	"""
    logging.debug('\nNous valors:\n%s', nous_ciclistes)
    df_nous_ciclistes, pred = nova_prediccio(nous_ciclistes, clustering_model)
    logging.info('\nPredicció dels valors per Toni Carballo EAC6:\n%s', pred)

    # Assignació dels nous valors als tipus
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info('Ciclista dorsal %s - Tipus: %s (cluster %s)',
                     df_nous_ciclistes.iloc[i]['dorsal'], t[0]['name'], p)

    # Generació d'informes
    generar_informes(ciclistes_data, tipus)

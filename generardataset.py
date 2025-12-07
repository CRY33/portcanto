import os
import logging
import numpy as np
"TONI CARBALLA EAC6"


def generar_dataset(num, ind, dicc):
    """
    Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
    TODO: completar arguments, return. num és el número de files/ciclistes a generar.
    ind és l'index/identificador/dorsal.
    """
    dataset = []

    for i in range(num):
        dorsal = ind + i
        tipus = dicc["name"]

        # Generar temps de pujada amb distribució normal
        temps_pujada = int(np.random.normal(dicc["mu_p"], dicc["sigma"]))

        # Generar temps de baixada amb distribució normal
        temps_baixada = int(np.random.normal(dicc["mu_b"], dicc["sigma"]))

        # Assegurar que els temps són positius
        temps_pujada = max(temps_pujada, 1)
        temps_baixada = max(temps_baixada, 1)

        dataset.append([dorsal, tipus, temps_pujada, temps_baixada])

    return dataset


if __name__ == "__main__":

    STR_CICLISTES = 'data/ciclistes.csv'

    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_BE = 3240  # mitjana temps pujada bons escaladors
    MU_P_ME = 4268  # mitjana temps pujada mals escaladors
    MU_B_BB = 1440  # mitjana temps baixada bons baixadors
    MU_B_MB = 2160  # mitjana temps baixada mals baixadors
    SIGMA = 240  # 240 s = 4 min
    DICC = [
        {"name": "BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    # Generar el dataset per a tots els tipus de ciclistes
    dataset_completo = []
    dorsal_actual = 1
    # Generar 100 ciclistes de cada tipus (total 400)
    for tipo_ciclista in DICC:
        dataset_tipo = generar_dataset(400, dorsal_actual, tipo_ciclista)
        dataset_completo.extend(dataset_tipo)
        dorsal_actual += 100

    # Guardar el dataset en CSV
    import csv
    os.makedirs('data', exist_ok=True)

    with open(STR_CICLISTES, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dorsal', 'tipus', 'temps_pujada', 'temps_baixada'])
        writer.writerows(dataset_completo)

    logging.info("s'ha generat data/ciclistes.csv")

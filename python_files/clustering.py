from imports import *
from preprocessing import *


def apply_kmeans(n_clusters,data):
    # Création d'un modèle KMeans avec le nombre de clusters spécifié
    kmeans=KMeans(n_clusters=n_clusters)
    # Entraînement du modèle sur les données fournies
    kmeans.fit(data)
    # Retourne le modèle entraîné
    return kmeans 


    
def clustering(df, n_parts=1, status_column="heat_on", n_clusters_list=None):
    """
    Applique le clustering K-means automatiquement en divisant les colonnes horaires en n_parts égales.
    
    - df : DataFrame avec 24 colonnes horaires (0 à 23)
    - n_parts : nombre de parties à créer (1 = pas de partition)
    - status_column : colonne  indiquant si le chauffage est activé
    - n_clusters_list : liste du nombre de clusters à appliquer par partie (taille doit être = n_parts)
    
    Retour : 
    - df avec colonnes "clusters_1", "clusters_2", ..., une pour chaque partie ou "clusters_1" si pas de partition
    - dict centroids_dict où chaque clé "clusters_i" a pour valeur un np.array des centroïdes (moyennes) des clusters
    """
    from sklearn.cluster import KMeans
    import numpy as np

    if n_clusters_list is None or len(n_clusters_list) != n_parts:
        raise ValueError("Tu dois fournir une liste n_clusters_list de même longueur que n_parts.")

    df_final = add_binary_column(df.copy(), column_name=status_column)
    df_final.columns = df_final.columns.astype(str)

    hour_columns = list(map(str, range(24)))
    step = 24 // n_parts
    parts_cols = [hour_columns[i * step: (i + 1) * step] for i in range(n_parts)]

    centroids_dict = {}

    for i, (n_clusters, cols) in enumerate(zip(n_clusters_list, parts_cols), start=1):
        df_part = df_final[cols + [status_column]].copy()
        df_heat = df_part[df_part[status_column] == 1].drop(columns=[status_column])
        
        model = apply_kmeans(n_clusters=n_clusters, data=df_heat)

        cluster_col = f"clusters_{i}"
        df_final.loc[df_part[status_column] == 1, cluster_col] = model.labels_
        df_final.loc[df_part[status_column] == 0, cluster_col] = n_clusters 

        # Calcul des centroïdes (moyennes) pour chaque cluster
        centroids = []
        for c in range(n_clusters):
            cluster_points = df_heat.iloc[model.labels_ == c]
            centroid = cluster_points.mean(axis=0).values
            centroids.append(centroid)
        centroids_dict[cluster_col] = np.array(centroids)

    return df_final, centroids_dict



def add_profil_and_status(input_df, conso_df, status_col="heat_on", profil_cols=None):
    """
    Ajoute la colonne 'status_col' et une ou plusieurs colonnes 'profil_cols' de 'conso_df' à 'input_df'.
    
    - input_df : DataFrame de base
    - conso_df : DataFrame contenant les colonnes à ajouter
    - status_col : colonne du statut (par défaut 'heat_on')
    - profil_cols : une chaîne (ex: "clusters_1") ou une liste (ex: ["clusters_1", "clusters_2", ...])
    """
    # Copie du DataFrame d'entrée pour éviter de modifier l'original
    df = input_df.copy()
    # Ajout de la colonne de statut
    df[status_col] = conso_df[status_col]

    # Si profil_cols est une chaîne unique, la transforme en liste pour homogénéiser le traitement
    if isinstance(profil_cols, str):
        profil_cols = [profil_cols]  

    # Ajoute les colonnes profil spécifiées si présentes
    if profil_cols:
        for col in profil_cols:
            df[col] = conso_df[col]

    # Conversion des noms de colonnes en string
    df.columns = df.columns.astype(str)
    # Retourne le DataFrame enrichi
    return df


def plot_clusters(consommation): 
    """
    Visualiser les séries temporelles de chaque cluster, avec les centroïdes marqués en rouge.
    """
    min_val = consommation.iloc[:, :-1].min().min() 
    max_val = consommation.iloc[:, :-1].max().max()
    ylim = [min_val - 2, max_val + 2]  
    unique_clusters = consommation["clusters_1"].unique()  
    num_clusters = len(unique_clusters) 
    num_points = consommation.shape[1] - 1  
    fig, axes = plt.subplots((num_clusters + 1) // 2, 2, figsize=(10, 10)) 
    axes = axes.flatten()
    for i, cluster in enumerate(unique_clusters): 
        ax = axes[i]  
        cluster_data = consommation[consommation["clusters_1"] == cluster]      
        for index, row in cluster_data.iterrows():
            ax.plot(range(num_points), row.iloc[:-1], color='gray', alpha=0.5)   
        center = cluster_data.iloc[:, :-1].mean(axis=0) 
        ax.plot(range(num_points), center, color='red', label=f'Cluster {cluster} ({len(cluster_data)})')
        ax.set_xlim([0, num_points])  
        ax.set_ylim(ylim) 
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel("Heures")
        ax.set_ylabel("Consommation (kJ/h)") 
        ax.legend()
    plt.tight_layout()
    plt.show()

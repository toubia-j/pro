from imports import *

def extract_columns(filepath, column_index=4):
    # Lire le fichier CSV avec tabulation
    df = pd.read_csv(filepath, delimiter="\t")
    # Extraire la colonne demandée
    values = df.iloc[:, column_index].values
    # Reshape en DataFrame 24 colonnes (24 heures)
    return pd.DataFrame(values.reshape(-1, 24))


def extract_and_concat_consommation(city_paths, column_index, prefix):
    """
    Extrait des colonnes spécifiques de plusieurs fichiers CSV correspondant à différentes villes,
    crée un DataFrame pour chaque ville nommé {prefix}{ville}, puis concatène tous les DataFrames
    dans un seul DataFrame global nommé df_combined_{prefix}.

    """
    extracted_data = []
    for city_name, path in city_paths.items():
        data = extract_columns(path, column_index)
        globals()[f"{prefix}{city_name}"] = data  # crée une variable globale avec le nom dynamique
        extracted_data.append(data)
    combined_df = pd.concat(extracted_data, axis=0).reset_index(drop=True)
    globals()[f"df_combined_{prefix}"] = combined_df  # variable globale pour le concaténé
    return combined_df


def add_binary_column(df, column_name="heat_on"):
    """
    Ajout d'une colonne binaire pour identifier les jours de consommation :
    - '1' indique un jour "ON" (consommation > 0)
    - '0' indique un jour "OFF" (consommation = 0)
    """
    df[column_name] = (df.drop(columns=[column_name], errors='ignore').sum(axis=1) > 0).astype(int)
    return df

def add_heating_season(df, date_column='Date'):
    """
    Ajoute une colonne 'heat_on' qui vaut 1 si la date est entre le 1er novembre et le 30 avril, sinon 0.

    """ 
    # Extraire le mois et le jour
    month_day = df[date_column].dt.month * 100 + df[date_column].dt.day
    
    # Appliquer la condition: 1 si entre 1101 (1er nov) et 0430 (30 avril), sinon 0
    df['heat'] = ((month_day >= 1101) | (month_day <= 430)).astype(int)
    
    return df



def extract_and_store_data(files, prefix, column_index):
    """
    Pour chaque fichier dans `files`, extrait toutes les colonnes
    et les stocke dans des variables globales nommées comme <NomColonne>_<ville>.
    """
    for city, path in files.items():
        data = extract_columns(path, column_index)  # Extraire les données pour chaque ville
        globals()[f"{prefix}{city}"] = data  # Stocker dans la variable globale



def extract_and_combine_all(city_groups, prefix_column_map):
    """
    Combine toutes les colonnes extraites (ayant le même préfixe) pour le groupe de villes actuel.
    Exemple : combine Text_agen, Text_albi, etc., en un seul DataFrame : Text_combined_toulouse.
    """
    combined_data = {}  # Dictionnaire pour stocker les DataFrames combinés

    # Parcourir les groupes de villes
    for group_name, files in city_groups.items():
        # Parcourir les préfixes et indices de colonnes
        for prefix, col_index in prefix_column_map.items():
            # Extraire et stocker les données
            extract_and_store_data(files, prefix, col_index)

            # Combiner les DataFrames pour chaque ville
            dfs = []
            for city in files.keys():
                var_name = f"{prefix}{city}"
                if var_name in globals():
                    dfs.append(globals()[var_name])  # Ajouter à la liste des DataFrames à combiner

            if dfs:  # Si des DataFrames existent à combiner
                combined_name = f"{prefix}combined_{group_name}"  # Nom du DataFrame combiné
                combined_data[combined_name] = pd.concat(dfs, axis=0).reset_index(drop=True)  # Combinaison des DataFrames

    return combined_data  # Retourner les DataFrames combinés


def make_column_names_unique(columns):
    """
    Cette fonction rend les noms des colonnes uniques en ajoutant un suffixe aux doublons.
    """
    seen = {}
    result = []
    for col in columns:
        # Si la colonne n'a pas encore été vue, l'ajouter telle quelle
        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            # Sinon, ajouter un suffixe pour la rendre unique
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result




def downsample_majority_class(df, target_column):
    """
    Réduire la classe majoritaire pour qu'elle soit égale au nombre de la classe maximale des autres classes.
    """
    # Compter le nombre d'exemples par classe
    counts = df[target_column].value_counts()
    # Trouver la classe majoritaire
    majority_value = counts.idxmax()
    # Trouver la taille maximale parmi les autres classes
    max_other = counts.drop(index=majority_value).max()
    # Sélectionner uniquement la classe majoritaire
    df_majority = df[df[target_column] == majority_value]
    # Sous-échantillonnage de la classe majoritaire pour l'équilibrer
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=max_other, random_state=42)
    # Sélectionner les autres classes
    df_others = df[df[target_column] != majority_value]
    # Concaténer et mélanger les données équilibrées
    balanced_df = pd.concat([df_majority_downsampled, df_others]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


    
def preprocess_data(Text_combined, clustering_heat, Test_Text_heat, name_combined):
    """
    -Cette fonction prépare les données pour un modèle LSTM.
    -L'équilibrage de la classe majoritaire est effectué uniquement sur les jours prédits, 
    et n'est pas effectué sur les jours passés utilisés comme entrées (t-1).
    -La prédiction est faite en fonction des différentes données d'entrée et de consommation,
    ainsi que du profil réel à t-1 et des différentes données d'entrée et des profils prédits à t.
    """
    # Calcul de l'indice de séparation (80% des données)
    split_index = int(0.8 * len(clustering_heat))
    # Copie du DataFrame de texte combiné
    df = Text_combined.copy()
    # Ajout de la colonne 'heat_on' issue du clustering
    df['heat_on'] = clustering_heat['heat_on']

    # Vérification des colonnes dupliquées dans df
    duplicates_df = df.columns[df.columns.duplicated()]
    # Vérification des colonnes dupliquées dans clustering_heat
    duplicates_clustering_heat = clustering_heat.columns[clustering_heat.columns.duplicated()]
    # Si doublons dans df, rendre les noms uniques
    if len(duplicates_df) > 0:
        df.columns = make_column_names_unique(df.columns)
    # Si doublons dans clustering_heat, rendre les noms uniques
    if len(duplicates_clustering_heat) > 0:
        clustering_heat.columns = make_column_names_unique(clustering_heat.columns)

    # Sélection des colonnes liées aux clusters
    cluster_cols = clustering_heat.filter(like='cluster').columns
    # Assignation des clusters sur la partie train (jours passés)
    df.loc[:split_index - 1, cluster_cols] = clustering_heat.loc[:split_index - 1, cluster_cols]

    # Sélection des colonnes de prédiction de clusters dans Test_Text_heat
    cluster_cols2 = Test_Text_heat.filter(like='y_pred_Gradient').columns
    # Pour chaque cluster prédit, ajout dans df pour la partie test (jours futurs)
    for cluster_idx in range(1, len(cluster_cols2) + 1):
        cluster_col_name = f'y_pred_Gradient Boosting_clusters_{cluster_idx}'
        df.loc[split_index:, f'clusters_{cluster_idx}'] = Test_Text_heat.loc[:, cluster_col_name].values

    # Rendre les noms de colonnes uniques dans df
    df.columns = make_column_names_unique(df.columns)

    # Ajout de colonnes supplémentaires pour l'équilibrage, avec un ID jour
    df = pd.concat([pd.Series(range(len(clustering_heat))), df, clustering_heat.iloc[:, :-(len(cluster_cols) + 1)]], axis=1).reset_index(drop=True)

    # Gestion des colonnes dupliquées après concaténation
    duplicates = df.columns[df.columns.duplicated()]
    df.columns = make_column_names_unique(df.columns)
    df.columns = df.columns.astype(str)

    # Application du downsampling sur la classe majoritaire ('heat_on')
    df2 = downsample_majority_class(df, 'heat_on')
    df2.columns = make_column_names_unique(df2.columns)

    # Calcul du nombre de blocs en fonction du nom de fichier
    n_blocks = len(name_combined.split('_combined')[0].split('_'))
    parts = name_combined.split('_combined')[0].split('_')
    formatted = ' and '.join(parts)
    print(f"Prediction based on : {formatted}")

    # Nombre de colonnes température (24h * nombre de blocs)
    n_temp_cols = 24 * n_blocks

    # Initialisation des scalers pour la température et la consommation
    scaler_temp = StandardScaler()
    scaler_cons = StandardScaler()

    # Sélection des colonnes clusters dans df et df2
    cluster_cols = df.columns[df.columns.str.contains('clusters_')]
    cluster_cols2 = df2.columns[df2.columns.str.contains('clusters_')]

    # Standardisation et assemblage des données pour df
    df_scaled = np.hstack([
        df.iloc[:, 0:1].values,  # ID du jour
        scaler_temp.fit_transform(df.iloc[:, 1:1 + n_temp_cols]),  # Température
        df.iloc[:, 1 + n_temp_cols:1 + n_temp_cols + 1].values,  # 'heat_on'
        df[cluster_cols].values,  # Clusters
        scaler_cons.fit_transform(df.iloc[:, -24:])  # Consommation
    ])

    # Standardisation et assemblage des données pour df2 (données équilibrées)
    df_scaled2 = np.hstack([
        df2.iloc[:, 0:1].values,  # ID du jour
        scaler_temp.fit_transform(df2.iloc[:, 1:1 + n_temp_cols]),  # Température
        df2.iloc[:, 1 + n_temp_cols:1 + n_temp_cols + 1].values,  # 'heat_on'
        df2[cluster_cols2].values,  # Clusters
        scaler_cons.fit_transform(df2.iloc[:, -24:])  # Consommation
    ])

    # Conversion des arrays numpy en DataFrames
    df_final = pd.DataFrame(df_scaled, columns=df.columns)
    df_final2 = pd.DataFrame(df_scaled2, columns=df2.columns)

    # Extraction des valeurs numpy
    data = df_final.values
    data2 = df_final2.values
    # Filtrage des jours avec ID non nul
    data2 = data2[data2[:, 0] != 0]

    # Initialisation des listes d'entrées et cibles
    X2, y2 = [], []
    for i in data2[:, 0]:
        # Données du jour précédent
        prev_data = data[data[:, 0] == i - 1, 1:]
        # Données du jour courant avec features sélectionnées
        current_data2 = data2[data2[:, 0] == i, 1:1 + n_temp_cols + len(cluster_cols) + 1]
        # Assemblage des données d'entrée (t-1 et t)
        X2.append(np.hstack([prev_data, current_data2]))
        # Cibles pour la prédiction
        y2.append(data2[data2[:, 0] == i, 1 + n_temp_cols + 1 + len(cluster_cols):])

    # Conversion en arrays numpy
    X2, y2 = np.array(X2), np.array(y2)
    # Reshape pour correspondre aux dimensions attendues par le modèle LSTM
    X2 = X2.reshape(X2.shape[0], X2.shape[2])
    y2 = y2.reshape(y2.shape[0], y2.shape[2])
    X2 = X2.reshape(X2.shape[0], 1, X2.shape[1])

    # Séparation train/test sur les données transformées
    idx_split = int((X2.shape[0] * 8) / 10)
    X_train2 = X2[:idx_split, :].astype(float)
    X_test2 = X2[idx_split:, :].astype(float)
    y_train2 = y2[:idx_split, :].astype(float)
    y_test2 = y2[idx_split:, :].astype(float)

    # Retourne les données train/test et les scalers
    return X_train2, X_test2, y_train2, y_test2, scaler_temp, scaler_cons


def histogramme_moyenne(df_variable, variable_name="Variable"):
    df = df_variable.copy()
    df['date'] = pd.to_datetime(df['Date'])
    df['year'] = df['date'].dt.year

    hourly_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    hourly_cols.remove('year')
    df['moyenne_journaliere'] = df[hourly_cols].mean(axis=1)

    years = sorted(df['year'].unique())

    # Plages fixes (3 premières) selon min et max global
    global_min = df['moyenne_journaliere'].min()
    global_max = df['moyenne_journaliere'].max()
    bins_fixed = np.linspace(global_min, global_max, 5)  # 4 segments = 5 bornes

    # Pour la légende : 
    # 3 premiers segments avec bornes fixes "de X à Y"
    # dernier segment dynamique, on affiche juste "≥ début_segment_4"
    legend_labels = [
        f"{bins_fixed[i]:.1f} – {bins_fixed[i+1]:.1f}" for i in range(3)
    ] + [f"≥ {bins_fixed[3]:.1f} (variable)"]

    data_percents = []
    labels = []

    for year in years:
        moyennes = df[df['year'] == year]['moyenne_journaliere'].dropna()
        max_year = moyennes.max()

        # Construire les bins pour cette année : 3 premiers fixes + dernier segment dynamique
        bins_year = list(bins_fixed[:4]) + [max_year]

        counts, _ = np.histogram(moyennes, bins=bins_year)
        total = counts.sum()
        percents = (counts / total * 100) if total > 0 else np.zeros(len(counts))

        data_percents.append(percents)
        labels.append(year)

    df_percents = pd.DataFrame(data_percents, columns=legend_labels, index=labels)

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    ax = df_percents.plot(kind='bar', stacked=True, color=colors, figsize=(12, 7))

    for i, percents in enumerate(data_percents):
        bottom = 0
        for j, pct in enumerate(percents):
            if pct > 0:
                ax.text(i, bottom + pct / 2, f"{pct:.1f}%", ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
                bottom += pct

    ax.set_xlabel("Année")
    ax.set_ylabel("Pourcentage de jours")
    ax.set_title(f"Distribution annuelle de {variable_name} en 4 plages (dernier segment dynamique)")
    ax.legend(title="Plages (3 fixes + dernier variable)", title_fontsize=10)

    plt.tight_layout()
    plt.show()

 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def histogramme_somme_journaliere(df_variable, variable_name="Variable"):
    df = df_variable.copy()
    df['date'] = pd.to_datetime(df['Date'])
    df['year'] = df['date'].dt.year

    # Sélection des colonnes numériques sauf 'year'
    hourly_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in hourly_cols:
        hourly_cols.remove('year')

    # Somme journalière (au lieu de moyenne)
    df['somme_journaliere'] = df[hourly_cols].sum(axis=1)

    years = sorted(df['year'].unique())

    # Plages fixes (3 premières) selon min et max global
    global_min = df['somme_journaliere'].min()
    global_max = df['somme_journaliere'].max()
    bins_fixed = np.linspace(global_min, global_max, 5)  # 4 segments = 5 bornes

    # Pour la légende : 3 premiers segments fixes + dernier dynamique
    legend_labels = [
        f"{bins_fixed[i]:.1f} – {bins_fixed[i+1]:.1f}" for i in range(3)
    ] + [f"≥ {bins_fixed[3]:.1f} (variable)"]

    data_percents = []
    labels = []

    for year in years:
        sommes = df[df['year'] == year]['somme_journaliere'].dropna()
        max_year = sommes.max()

        # S'assurer que max_year est >= dernier bord fixe pour éviter erreur bins
        if max_year < bins_fixed[3]:
            max_year = bins_fixed[3]

        bins_year = list(bins_fixed[:4]) + [max_year]

        counts, _ = np.histogram(sommes, bins=bins_year)
        total = counts.sum()
        percents = (counts / total * 100) if total > 0 else np.zeros(len(counts))

        data_percents.append(percents)
        labels.append(year)

    df_percents = pd.DataFrame(data_percents, columns=legend_labels, index=labels)

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    ax = df_percents.plot(kind='bar', stacked=True, color=colors, figsize=(12, 7))

    for i, percents in enumerate(data_percents):
        bottom = 0
        for j, pct in enumerate(percents):
            if pct > 0:
                ax.text(i, bottom + pct / 2, f"{pct:.1f}%", ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
                bottom += pct

    ax.set_xlabel("Année")
    ax.set_ylabel("Pourcentage de jours")
    ax.set_title(f"Distribution annuelle de {variable_name} en 4 plages (dernier segment dynamique)")
    ax.legend(title="Plages (3 fixes + dernier variable)", title_fontsize=10)

    plt.tight_layout()
    plt.show()

    return df_percents

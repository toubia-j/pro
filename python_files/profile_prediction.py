from imports import *
from preprocessing import *


def balance_clusters(X, y):
    """
    Équilibre les clusters en ajustant le cluster majoritaire  pour avoir le même nombre d'exemples 
    que le cluster le plus grand parmi les autres.
    """
    
    df = X.copy()
    # Ajoute la colonne des clusters au DataFrame
    df['clusters'] = y
    # Affiche les colonnes après ajout (debug)
    print(df.columns)
    # Groupe les données par cluster
    cluster_groups = df.groupby('clusters')
    # Compte le nombre d'exemples par cluster
    cluster_counts = df['clusters'].value_counts()
    # Trouve la taille maximale des clusters sauf le cluster 3.0 (cluster majoritaire)
    max_other_clusters = cluster_counts[cluster_counts.index != 3.0].max()
    # Récupère le groupe du cluster 3.0
    cluster_3 = cluster_groups.get_group(3.0)
    # Échantillonne au hasard dans le cluster 3.0 pour équilibrer la taille
    cluster_3_resampled = cluster_3.sample(n=max_other_clusters, random_state=42)
    # Concatène tous les clusters sauf 3.0 avec le cluster 3.0 rééchantillonné
    balanced_df = pd.concat([cluster_groups.get_group(cluster) for cluster in cluster_counts.index if cluster != 3.0] + [cluster_3_resampled])
    # Mélange les données pour éviter un ordre biaisé
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Sépare les features des labels
    X_balanced = balanced_df.drop(columns=["clusters"])
    y_balanced = balanced_df["clusters"]
    return X_balanced, y_balanced
    


def evaluate_models_split(df, target_cols, models, split_ratio=8):
    """
    Évalue plusieurs modèles (mono-label ou multi-label) avec séparation manuelle (80% par défaut).
    Si `target_cols` contient plusieurs colonnes => multi-label.
    Retourne :
      - un dictionnaire avec les métriques,
      - un DataFrame avec les vraies valeurs et prédictions.
    """
    # Détecte si on est en multi-label
    multi_label = isinstance(target_cols, list) and len(target_cols) > 1
    # Sépare les colonnes cibles selon le type multi ou mono-label
    y = df[target_cols] if multi_label else df[[target_cols]]
    X = df.drop(columns=target_cols if multi_label else [target_cols])  
    
    # Calcule l'indice de séparation en fonction du ratio (par ex. 80%)
    split_index = int((X.shape[0] * split_ratio) / 10)
    # Divise en train/test
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    results = {}
    # Copie X_test pour y ajouter les résultats
    df_test_results = X_test.copy()
    # Ajoute les vraies valeurs dans df_test_results pour comparaison
    for col in (target_cols if multi_label else [target_cols]):
        df_test_results[f'y_true_{col}'] = y_test[col].values

    if multi_label:
        # Binarise les labels multi-label pour l'entraînement
        mlb = MultiLabelBinarizer()
        y_train_bin = mlb.fit_transform(y_train.values.tolist())
        y_test_bin = mlb.transform(y_test.values.tolist())

    for name, model in models.items():
        print(f"\nÉvaluation de {name}...")
        start_time = time.time()
        if multi_label:
            # Entraîne et prédit pour multi-label
            model.fit(X_train, y_train_bin)
            y_pred_bin = model.predict(X_test)
        else:
            # Entraîne et prédit pour mono-label
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)
        exec_time = time.time() - start_time

        if multi_label:
            # Calcul des scores multi-label
            f1 = f1_score(y_test_bin, y_pred_bin, average='weighted')
            acc = accuracy_score(y_test_bin, y_pred_bin)
            zero_one = zero_one_loss(y_test_bin, y_pred_bin)
            hamming = hamming_loss(y_test_bin, y_pred_bin)

            # Stocke les métriques dans le dictionnaire
            results[name] = {
                "f1_score": f1,
                "accuracy": acc,
                "zero_one_loss": zero_one,
                "hamming_loss": hamming,
                "execution_time (s)": exec_time
            }

            # Ajoute les prédictions binaires dans df_test_results pour chaque cible
            for i, col in enumerate(target_cols):
                df_test_results[f'y_pred_{name}_{col}'] = y_pred_bin[:, i]

            print(f"{name} - F1: {f1:.4f} - Accuracy: {acc:.4f} - 0/1 Loss: {zero_one:.4f} - Hamming Loss: {hamming:.4f} - Temps: {exec_time:.2f}s")
        else:
            # Calcul des scores mono-label
            f1 = f1_score(y_test, y_pred, average='weighted')
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Stocke les métriques
            results[name] = {
                "f1_score": f1,
                "accuracy": acc,
                "execution_time (s)": exec_time
            }

            # Ajoute les prédictions dans df_test_results
            df_test_results[f'y_pred_{name}_clusters_1'] = y_pred

            # Affiche la matrice de confusion
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel('Prédictions')
            plt.ylabel('Vraies classes')
            plt.title(f'Matrice de confusion - {name}')
            plt.show()

            print(f"{name} - F1: {f1:.4f} - Accuracy: {acc:.4f} - Temps: {exec_time:.2f}s")
        print("###################################################################")

    return results, df_test_results

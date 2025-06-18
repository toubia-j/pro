# 🔥 Prédiction de la consommation de chauffage

Ce projet a pour but d’analyser, profiler et prédire la consommation de chauffage à partir de données météo et des comportements observés.

Le processus se déroule en plusieurs étapes :

1. **Identification des profils de consommation de chauffage** :  
   Analyse des données pour détecter des profils types de consommation.

2. **Prédiction par profil** :  
   Prédiction de la consommation pour chaque profil identifié en fonction de différentes grandeurs (température, occupation, etc.).

3. **Prédiction sur 24 heures** :  
   Prédiction complète de la consommation sur une période de 24 heures, basée sur les profils prédits et les données météo.


---

# 📁 Structure du projet

### <span style="color:red;">📄 requirements  </span>
 
  Contient le fichier `requirements.txt` avec les dépendances Python nécessaires.  
  **<span style="color:green;">Installation : </span>** 
  ```bash
  python -m pip install -r requirements/requirements.txt
  ```
---

### <span style="color:red;">📂 python_files  </span>
Contient les fonctions Python utilisées dans les notebooks (extraction, traitement, modélisation, etc.).

---


### <span style="color:red;">📂 analyse  </span>
Notebooks d’analyse exploratoire, notamment l’étude des corrélations entre variables.

---

### <span style="color:red;">📂 clustering  </span>

Identification des profils types de consommation de chauffage. Ces profils servent ensuite à faire des prédictions personnalisées selon le comportement détecté.

**<span style="color:green;">Notebooks à exécuter</span>** : ceux terminant par `_vector_centroide`.

---

### <span style="color:red;">📂 prediction</span>

#### profile_prediction

Prédictions de consommation par profil, en fonction d’une grandeur :  
- `Text` → température extérieure  
- `Text_occupation` → température + taux d’occupation  
- etc.

**<span style="color:green;">Notebooks à exécuter</span>** : le notebook correspondant à la grandeur, par exemple `..._Text_vector_centroide.ipynb`.

#### 24hours_prediction
Prédictions de 24 heures, selon des grandeurs (d) et les profils définis dans `profile_prediction`, basées sur les données du jour précédent et du jour en question..

**<span style="color:green;">Notebooks à exécuter</span>** : `..._vector_centroide.ipynb` avec la même grandeur.

---

# ✅ Ordre d’exécution recommandé

1. `notebook/clustering/` → notebooks terminant par (`_vector_centroide.ipynb`) (ex. : heating_conso_7years_data_from_1_novembre_to_31_avril_vector_centroide.ipynb)
2. `notebook/prediction/profile_prediction/` → notebooks pour la grandeur choisie (`..._vector_centroide.ipynb`) (ex. : profile_prediction_based_on_Text_7years_data_from_1_novembre_to_31_avril_vector_centroide.ipynb)
  
3. `notebook/prediction/24hours_prediction/` → notebooks avec la même grandeur (`..._vector_centroide.ipynb`) (ex. : 24hours_prediction_based_on_Text_7years_data_from_1_novembre_to_31_avril_vector_centroide.ipynb)  

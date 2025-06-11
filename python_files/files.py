from imports import *


toulouse = {
    "agen": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Agen/Simulation_Outputs",
    "albi": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Albi/Simulation_Outputs",
    "auch": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Auch/Simulation_Outputs",
    "toulouse": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Toulouse/Simulation_Outputs"
}

zurich= {    
    "Birmensdorf":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Birmensdorf/Simulation_Outputs",
    "Taenikon":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Taenikon/Simulation_Outputs",
    "Zurich_fluntern":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_Fluntern/Simulation_Outputs",
    "Zurich_kloten":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_kloten/Simulation_Outputs"
}

seville = {
    "Cordoba": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Cordoba/Simulation_Outputs",
    "Granada": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Granada/Simulation_Outputs",
    "Malaga": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Malaga/Simulation_Outputs",
    "Sevilla": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Sevilla/Simulation_Outputs"   
}

toulouse_meteo = {
    "agen": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Agen/Meteo_input",
    "albi": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Albi/Meteo_input",
    "auch": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Auch/Meteo_input",
    "toulouse": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Toulouse/Meteo_input"
}

zurich_meteo = {
    "Birmensdorf":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Birmensdorf/Meteo_input",
    "Taenikon":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Taenikon/Meteo_input",
    "Zurich_fluntern":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_Fluntern/Meteo_input",
    "Zurich_kloten":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_kloten/Meteo_input",
}

seville_meteo = {
    "Cordoba": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Cordoba/Meteo_input",
    "Granada": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Granada/Meteo_input",
    "Malaga": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Malaga/Meteo_input",
    "Sevilla": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Sevilla/Meteo_input"  
}





Simulation_Dataset = {
    "2018": "../data/simulated_data_7years/Mal_isole/2018/Simulation_Outputs",
    "2019": "../data/simulated_data_7years/Mal_isole/2019/Simulation_Outputs",
    "2020": "../data/simulated_data_7years/Mal_isole/2020/Simulation_Outputs",
    "2021": "../data/simulated_data_7years/Mal_isole/2021/Simulation_Outputs",
    "2022": "../data/simulated_data_7years/Mal_isole/2022/Simulation_Outputs",
    "2023": "../data/simulated_data_7years/Mal_isole/2023/Simulation_Outputs",
    "2024": "../data/simulated_data_7years/Mal_isole/2024/Simulation_Outputs"   
}

Weather_Dataset = {
    "2018": "../data/simulated_data_7years/Mal_isole/2018/Weather_Dataset.txt",
    "2019": "../data/simulated_data_7years/Mal_isole/2019/Weather_Dataset.txt",
    "2020": "../data/simulated_data_7years/Mal_isole/2020/Weather_Dataset.txt",
    "2021": "../data/simulated_data_7years/Mal_isole/2021/Weather_Dataset.txt",
    "2022": "../data/simulated_data_7years/Mal_isole/2022/Weather_Dataset.txt",
    "2023": "../data/simulated_data_7years/Mal_isole/2023/Weather_Dataset.txt",
    "2024": "../data/simulated_data_7years/Mal_isole/2024/Weather_Dataset.txt"   
}



models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVC": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


models2 = {
    "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Logistic Regression": MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=1000)),
    "SVC": MultiOutputClassifier(SVC(random_state=42)),
    "KNN": MultiOutputClassifier(KNeighborsClassifier()),
    "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
}
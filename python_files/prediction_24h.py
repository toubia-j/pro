from imports import *
from preprocessing import *




def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler_temp, scaler_cons):
    # Compilation du modèle avec optimizer, loss et métriques
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    # Callback early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Séparation d'un ensemble validation à partir des données d'entraînement (sans shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
    # Entraînement du modèle avec validation et early stopping
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Affichage de la courbe de loss durant l'entraînement
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss during training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
   
    # Évaluation du modèle sur l'ensemble validation
    loss, mae, mse = model.evaluate(X_val, y_val)
    rmse = np.sqrt(mse) 
    print(f"Validation Loss : {loss}")
    print(f"Validation MAE: {mae}")
    print(f"Validation MSE: {mse}")
    print(f"Validation RMSE: {rmse}")
    
    # Prédiction sur l'ensemble test
    predictions = model.predict(X_test) 
    y_test_reshape = y_test.reshape(-1, 24) 
    # Inverse transformation de la normalisation de la consommation
    predictions_norm = scaler_cons.inverse_transform(predictions)
    y_test_reshape_norm = scaler_cons.inverse_transform(y_test_reshape)

    # Calcul des métriques sur l'ensemble test
    mae_test = mean_absolute_error(y_test_reshape_norm, predictions_norm)
    mse_test = mean_squared_error(y_test_reshape_norm, predictions_norm)
    rmse_test = np.sqrt(mse_test)
    r2 = r2_score(y_test_reshape_norm, predictions_norm)
    cvrmse = rmse_test / np.mean(y_test_reshape_norm)

    print(f"Test MAE: {mae_test}")
    print(f"Test MSE: {mse_test}")
    print(f"Test RMSE: {rmse_test}")
    print(f"Test R²: {r2}")
    print(f"Test CVRMSE: {cvrmse}")

    return history, loss, mae, mse, rmse, mae_test, mse_test, rmse_test, r2, cvrmse, predictions



def model_lstm(X_train2, y_train2, X_test2, y_test2, scaler_temp, scaler_cons):
    # Définition du modèle LSTM séquentiel
    model2 = Sequential()
    # Couche LSTM avec 68 unités, activation tanh, forme d'entrée adaptée
    model2.add(LSTM(68, activation='tanh', input_shape=(X_train2.shape[1], X_train2.shape[2])))  
    # Couche Dropout pour régularisation
    model2.add(Dropout(0.2))
    # Couche Dense finale avec 24 sorties et activation linéaire (prédiction continue)
    model2.add(Dense(24, activation='linear'))   
    
    # Entraînement et évaluation du modèle
    history2, loss2, mae2, mse2, rmse2, mae_test2, mse_test2, rmse_test2, r2, cvrmse, predictions2 = train_and_evaluate(
        model2, X_train2, y_train2, X_test2, y_test2, scaler_temp, scaler_cons
    )
    
    return model2, history2, loss2, mae2, mse2, rmse2, mae_test2, mse_test2, rmse_test2, r2, cvrmse, predictions2





def create_line_gif_point_by_point_indices(
    indices,         # liste des indices des jours à tracer
    y_test,
    predictions,
    scaler_cons,
    filename="line_point_by_point.gif"
):
    """
    Crée un GIF où, pour chaque exemple (indice donné dans indices),
    les points vrais et prédits apparaissent un à un en ligne.
    """
    num_examples = len(indices)
    
    # On récupère les données rescalées pour les indices donnés
    y_test_rescaled = scaler_cons.inverse_transform(y_test[indices])
    predictions_rescaled = scaler_cons.inverse_transform(predictions[indices])

    fig, ax = plt.subplots(figsize=(10, 6))

    length = y_test_rescaled.shape[1]
    total_frames = num_examples * length

    def update(frame):
        ax.clear()

        example_idx = frame // length
        point_idx = frame % length

        y_true = y_test_rescaled[example_idx][:point_idx + 1]
        y_pred = predictions_rescaled[example_idx][:point_idx + 1]
        indices_x = np.arange(point_idx + 1)

        ax.plot(indices_x, y_true, marker='o', color='blue', label="Valeurs réelles", linestyle='-')
        ax.plot(indices_x, y_pred, marker='x', color='orange', label="Prédictions", linestyle='--')

        ax.set_ylabel("Consommation")
        ax.set_xlabel("Heures")
        ax.set_title(f"Différence entre valeur réelle et valeur prédite de consommation d'énergie - heure {point_idx + 1} sur {length}")
        ax.set_xticks(np.arange(length))
        ax.set_xticklabels([f"Heure {j}" for j in range(length)], rotation=45)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        y_max = max(y_test_rescaled[example_idx].max(), predictions_rescaled[example_idx].max())
        #ax.set_ylim(0, y_max * 1.1)
        ax.set_ylim(0, 5000)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, repeat=False)
    ani.save(filename, writer='pillow', fps=2)  # fps=2 pour voir doucement
    plt.close()


import plotly.graph_objects as go

def plot_true_vs_predicted_interactive(indices, y_test, predictions, scaler_cons):
    """
    Affiche un graphique interactif comparant les valeurs réelles et prédites pour plusieurs jours de test.
    """
    true_total = []
    pred_total = []
    x_ticks = []
    labels = []

    for i, idx in enumerate(indices):
        conso_reel = scaler_cons.inverse_transform(y_test[idx].reshape(1, -1))
        conso_pred = scaler_cons.inverse_transform(predictions[idx].reshape(1, -1))
        true_total.extend(conso_reel.flatten())
        pred_total.extend(conso_pred.flatten())
        x_ticks.extend(list(range(i * 24, (i + 1) * 24)))
        labels.extend([f"Jour {i+1} - h{j}" for j in range(24)])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_ticks,
        y=true_total,
        mode='lines+markers',
        name='Valeurs Réelles',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=x_ticks,
        y=pred_total,
        mode='lines+markers',
        name='Prédictions',
        line=dict(color='orange')
    ))

    # Ajout des lignes verticales entre les jours
    for i in range(1, len(indices)):
        fig.add_vline(x=i * 24, line=dict(color='gray', dash='dash'), opacity=0.3)

    fig.update_layout(
        title="Valeurs Réelles vs Prédictions (Interactif)",
        xaxis_title="Temps (heures concaténées)",
        yaxis_title="Consommation de chauffage",
        hovermode="x unified",
        xaxis=dict(tickmode='array', tickvals=x_ticks[::4], ticktext=labels[::4]),
        legend=dict(x=0.01, y=0.99)
    )

    fig.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Carregar dataset
def main():
    
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv", sep=',')
    print("Colunas do dataset:", df.columns)

    # Definir variáveis independentes (X) e alvo (y)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Criar modelo
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treinar modelo
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Avaliar o modelo
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print("Relatório de classificação:\n", classification_report(y_test, y_pred))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

    # Plotar acurácia
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title("Acurácia durante o treinamento")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

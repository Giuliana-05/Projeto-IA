import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

sns.set_style('whitegrid')

@st.cache_data
def load_data():
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
    return df

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(history.history['accuracy'], label='Treino')
    axs[0].plot(history.history['val_accuracy'], label='Validação')
    axs[0].set_title('Acurácia durante o treinamento')
    axs[0].set_xlabel('Época')
    axs[0].set_ylabel('Acurácia')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Treino')
    axs[1].plot(history.history['val_loss'], label='Validação')
    axs[1].set_title('Perda durante o treinamento')
    axs[1].set_xlabel('Época')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    st.pyplot(fig)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

def main():
    st.title("Projeto IA: Previsão de Doença Cardiovascular")
    st.write("Este projeto usa um modelo de rede neural para classificar pacientes com risco cardiovascular.")

    df = load_data()
    st.write("### Dados carregados:")
    st.dataframe(df.head())

    X = df.drop("target", axis=1)
    y = df["target"]

    if st.button("Treinar modelo"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = build_model(X_train.shape[1])

        with st.spinner('Treinando o modelo...'):
            history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

        st.success('Treinamento concluído!')

        y_pred = (model.predict(X_test) > 0.5).astype(int)

        st.write("### Relatório de classificação:")
        st.text(classification_report(y_test, y_pred))

        plot_training_history(history)
        plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()


   
    

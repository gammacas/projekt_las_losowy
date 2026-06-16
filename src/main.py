import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier  # Ta linijka naprawia Twój NameError

# Importujemy nasze własne moduły lokalne
from data_handler import load_and_preprocess_data, URL_TRAIN, URL_TEST
from models import train_baseline_model, train_tuned_random_forest

def evaluate_pipeline(model_name: str, model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:

    
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    predictions_custom = (probabilities >= threshold).astype(int)
    
    print(f"\n--- Wyniki ewaluacji dla: {model_name} (Próg: {threshold*100}%) ---")
    print(f"Dokładność (Accuracy):        {accuracy_score(y_test, predictions_custom):.4f}")
    print(f"Precyzja (Precision):         {precision_score(y_test, predictions_custom):.4f}")
    print(f"Czułość / Pełność (Recall):   {recall_score(y_test, predictions_custom):.4f}")
    print(f"F1-Score:                     {f1_score(y_test, predictions_custom):.4f}")

def plot_results(baseline_model, rf_model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    print("\n[4/4] Generowanie wizualizacji ")
    
    threshold = 0.3
    probabilities = rf_model.predict_proba(X_test)[:, 1]
    rf_predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, rf_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ruch Normalny', 'Cyberatak'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Macierz Błędów - Zoptymalizowany Las Losowy")
    plt.show()

    importances = rf_model.feature_importances_
    forest_importances = pd.Series(importances, index=X_test.columns).sort_values(ascending=False)
    top_10_features = forest_importances.head(10)

    plt.figure(figsize=(10, 6))
    top_10_features.plot.bar(color='teal')
    plt.title("Top 10 kluczowych cech ruchu sieciowego (Random Forest)")
    plt.ylabel("Stopień ważności cechy (Gini Impurity)")
    plt.xlabel("Metryki sieciowe")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    

    print("Generowanie porównawczej Krzywej ROC...")
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_estimator(baseline_model, X_test, y_test, color='blue', name='Drzewo Decyzyjne', ax=ax)
    

    RocCurveDisplay.from_estimator(rf_model, X_test, y_test, color='darkorange', name='Zoptymalizowany Las Losowy', ax=ax)
    

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Losowe zgadywanie (Baseline)')
    plt.title("Porównawcza Krzywa ROC - Drzewo vs Las")
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data(URL_TRAIN, URL_TEST)
    

    
    print("\n--- Rozkład klas w zbiorach ---")
    print("Zbiór TRENINGOWY (Materiały do nauki):")
    print(y_train.value_counts().rename({0: 'Ruch Normalny', 1: 'Cyberatak'}))
    print("\nZbiór TESTOWY (Pytania na egzaminie):")
    print(y_test.value_counts().rename({0: 'Ruch Normalny', 1: 'Cyberatak'}))
    print("-------------------------------\n")
    
    baseline_tree = train_baseline_model(X_train, y_train)
    optimized_rf = train_tuned_random_forest(X_train, y_train)
    
    evaluate_pipeline("Drzewo Decyzyjne (Baseline)", baseline_tree, X_test, y_test)
    evaluate_pipeline("Zoptymalizowany Las Losowy", optimized_rf, X_test, y_test)
    
    plot_results(baseline_tree,optimized_rf, X_test, y_test)
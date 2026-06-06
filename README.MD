# Wykrywanie anomalii w ruchu sieciowym metodą losowych lasów (Random Forest)

Implementacja systemu wykrywania intruzów (IDS) z wykorzystaniem klasyfikatora Random Forest, przetestowana na  zbiorze danych NSL-KDD.

## Wymagania
* Python 3.8+

## Instalacja
1. Skopiuj repozytorium na dysk lokalny.
2. Otwórz konsolę w folderze głównym projektu.
3. Utwórz wirtualne środowisko: python -m venv venv
4. Aktywuj środowisko: venv\Scripts\activate (Windows)
5. Zainstaluj pakiety: pip install -r requirements.txt

## Użytkowanie
1. Otwórz notatnik Jupyter wpisując w konsoli: jupyter notebook
2. Przejdź do folderu src/ i uruchom plik z kodem projektu.
3. Kod automatycznie pobierze zbiór NSL-KDD, przetworzy dane i wygeneruje metryki modelu oraz wykresy ewaluacyjne.
import csv
import joblib
import numpy as np
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

def main():
    print("Carregando dataset...")
    X = []
    y = []
    with open('fake_recogna_limpo.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append(row.get('texto_limpo', ''))
            y.append(int(row.get('Classe', 0)))

    print("Carregando modelo original (Pipeline)...")
    pipeline = joblib.load('models/best_linearsvc_model.joblib')

    print("Calibrando...")
    fe = FrozenEstimator(pipeline)
    calibrator = CalibratedClassifierCV(estimator=fe, method='sigmoid')
    calibrator.fit(X, y)

    print("Salvando modelo calibrado...")
    joblib.dump(calibrator, 'models/calibrated_pipeline.joblib')
    print("Pronto!")

if __name__ == '__main__':
    main()

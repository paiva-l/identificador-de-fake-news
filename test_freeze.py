from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import numpy as np

X = np.random.randn(10, 2)
y = np.random.randint(0, 2, 10)

svc = LinearSVC().fit(X, y)
fe = FrozenEstimator(svc)
cal = CalibratedClassifierCV(estimator=fe)
cal.fit(X, y)
print(cal.predict_proba(X))

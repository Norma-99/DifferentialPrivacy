import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

a = np.array([-1, -1, -1, -1, -1, 1, -1, -1, -1, -1])
b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])

print(mutual_info_classif(a.reshape(0,1), b, discrete_features = True)) # mutual information of 0.69, expressed in nats
print(mutual_info_score(a,b)) # information gain of 0.69, expressed in nats

# We can do it with mutual_info_classif or with mutual_info_score (I prefer mutual_info_score)
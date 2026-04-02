import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC

df = pd.read_csv("data/wesad_small.csv")

# much smaller sample for quantum
df_q = (
    df.groupby("emotion", group_keys=False)
      .apply(lambda x: x.sample(n=100, random_state=42))
      .reset_index(drop=True)
)

X = df_q[["ecg", "eda", "temp", "resp"]]
y = df_q["emotion"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
model = QSVC(feature_map=feature_map)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("QSVM Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=encoder.classes_))

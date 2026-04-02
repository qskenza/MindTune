import os
import pickle
import numpy as np
import pandas as pd

base_path = "data/WESAD"
subjects = [d for d in os.listdir(base_path) if d.startswith("S")]

rows = []

for subject in subjects:
    file_path = os.path.join(base_path, subject, f"{subject}.pkl")

    if not os.path.exists(file_path):
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    labels = np.array(data["label"]).flatten()
    chest = data["signal"]["chest"]

    ecg = np.array(chest["ECG"]).flatten()
    eda = np.array(chest["EDA"]).flatten()
    temp = np.array(chest["Temp"]).flatten()
    resp = np.array(chest["Resp"]).flatten()

    min_len = min(len(labels), len(ecg), len(eda), len(temp), len(resp))

    for i in range(min_len):
        label = labels[i]

        if label in [1, 2, 3]:
            if label == 1:
                emotion = "stress"
            elif label == 2:
                emotion = "happy"
            else:
                emotion = "calm"

            rows.append({
                "subject": subject,
                "ecg": float(ecg[i]),
                "eda": float(eda[i]),
                "temp": float(temp[i]),
                "resp": float(resp[i]),
                "emotion": emotion
            })

df = pd.DataFrame(rows)
df.to_csv("data/wesad_emotion_dataset.csv", index=False)

print("Dataset created")
print(df.head())
print(df["emotion"].value_counts())
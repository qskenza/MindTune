import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "heart_rate": np.random.randint(60, 110, 300),
    "hrv": np.random.randint(20, 80, 300),
    "activity": np.random.rand(300)
}

df = pd.DataFrame(data)

def label_emotion(row):
    if row["heart_rate"] > 95:
        return "stress"
    elif row["hrv"] > 55:
        return "calm"
    else:
        return "neutral"

df["emotion"] = df.apply(label_emotion, axis=1)

df.to_csv("emotion_dataset.csv", index=False)

print("Dataset created successfully")
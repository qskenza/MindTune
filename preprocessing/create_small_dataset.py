import pandas as pd

df = pd.read_csv("data/wesad_emotion_dataset.csv")

df_small = (
    df.groupby("emotion", group_keys=False)
      .apply(lambda x: x.sample(n=5000, random_state=42))
      .reset_index(drop=True)
)

df_small.to_csv("data/wesad_small.csv", index=False)

print("Small dataset created")
print(df_small["emotion"].value_counts())

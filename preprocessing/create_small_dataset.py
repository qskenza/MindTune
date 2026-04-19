import pandas as pd

# Load full processed dataset
df = pd.read_csv("data/wesad_emotion_dataset.csv")

print("Original shape:", df.shape)
print(df["emotion"].value_counts())

# -------- Option 1: Binary task for QSVM ----------
# stress vs non_stress

stress_labels = ["stress", "stressed"]
non_stress_labels = ["calm", "neutral", "happy"]

df["emotion"] = df["emotion"].str.lower()

df = df[df["emotion"].isin(stress_labels + non_stress_labels)].copy()

df["emotion"] = df["emotion"].apply(
    lambda x: "stress" if x in stress_labels else "non_stress"
)

# -------- Balanced small subset ----------
N_PER_CLASS = 300   # try 200, 300, 500 max first

df_small = (
    df.groupby("emotion", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), N_PER_CLASS), random_state=42))
      .reset_index(drop=True)
)

# Shuffle
df_small = df_small.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_small.to_csv("data/wesad_small_qsvm.csv", index=False)

print("\nSmall dataset created.")
print("Shape:", df_small.shape)
print(df_small["emotion"].value_counts())
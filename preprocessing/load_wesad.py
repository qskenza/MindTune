import pickle

file_path = "data/WESAD/S14/S14.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Top keys:", data.keys())
print("Signal keys:", data["signal"].keys())
print("First labels:", data["label"][:10])

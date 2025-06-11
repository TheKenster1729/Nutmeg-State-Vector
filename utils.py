import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

full_dataset = pd.read_csv("full_dataset.csv")

print(full_dataset)

test_set = pd.read_csv("test_set_streets.csv")

print(test_set)

train_set = pd.read_csv("training_set_streets.csv")

print(train_set)
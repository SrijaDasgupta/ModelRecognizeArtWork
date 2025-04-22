"""Data Analysis for Augmentation
"""
import pandas as pd
df = pd.read_csv('train.csv')

print(df.head(10))

print(df.shape)

print(df['attribute_ids'])

print(df['attribute_ids'].shape)

attributes = []
for row in df['attribute_ids']:
  attributes.append(row.split(" "))

print(len(attributes))

print(attributes[:10])

multiculture = []
for row in attributes:
  count = len([i for i in row if int(i) < 397])
  if count > 1:
    multiculture.append(row)

print(len(multiculture))

print(len(multiculture)/len(df['attribute_ids']) * 100)

multicultureNoTag = []
for row in attributes:
  countCulture = len([i for i in row if int(i) <= 397])
  countTag = len([i for i in row if int(i) > 397])
  if countCulture == countTag:
    multicultureNoTag.append(row)

print(len(multicultureNoTag))

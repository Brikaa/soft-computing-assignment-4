"""
- Read excel
- Choose random rows to be testing
- Rest of rows are training
"""
import pandas as pd
import random

test_indices = set()
excel = pd.read_excel('../concrete_data.xlsx')
no_tests = 175
data_size = 699
while len(test_indices) < no_tests:
  gen = random.randint(0, data_size)
  while gen in test_indices:
    gen = random.randint(0, data_size)
  test_indices.add(gen)
test_rows = []
training_rows = []
i = 0
for row in excel.iterrows():
  if i in test_indices:
    test_rows.append(row)
  else:
    training_rows.append(row)
  i += 1

for i in range(2):
  target = test_rows if i == 0 else training_rows
  print(len(target))
  for row in target:
    print(f"{row[1]['cement']} {row[1]['water']} {row[1]['superplasticizer']} {row[1]['age']} {row[1]['concrete_compressive_strength']}")

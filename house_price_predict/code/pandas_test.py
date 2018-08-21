import pandas as pd

data = [
    ['red', 'big', 2, True],
    ['blue', 'big', 1, True],
    ['blue', 'small', 3, False],
    ['yellow', 'big', 4, False]
]

df = pd.DataFrame(data)
print(df)

print('-' * 20)

print(pd.get_dummies(df))

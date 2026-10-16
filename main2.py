import pandas as pd

df = pd.DataFrame({
    "age": [22, 35, 48, 25, 41, 30],
    "income": [35000, 120000, 90000, 40000, 150000, 70000],
    "loan_amount": [500000, 200000, 800000, 600000, 300000, 400000],
    "loan_term": [60, 24, 120, 84, 36, 48],
    "default": [1, 0, 0, 1, 0, 0]
})

print(df)

y = df["default"]
print(y)



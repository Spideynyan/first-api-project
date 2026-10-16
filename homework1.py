import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


df = pd.DataFrame({
    "links_count": [
        0, 1, 8, 2, 5,
        2, 4, 1, 0, 3,
        3, 0, 6, 2, 1,
        5, 1, 4, 0, 7
    ],

    "words_count": [
        120, 250, 180, 90, 95,
        300, 140, 220, 150, 75,
        110, 280, 190, 200, 170,
        160, 240, 100, 310, 85
    ],

    "suspicious_words": [
        0, 2, 3, 1, 4,
        1, 2, 0, 0, 3,
        2, 0, 4, 1, 2,
        2, 1, 3, 0, 5
    ],

    "sender_known": [
        1, 1, 0, 1, 0,
        1, 0, 1, 1, 0,
        0, 1, 0, 1, 1,
        0, 1, 0, 1, 0
    ],

    "spam": [
        0, 1, 0, 0, 1,
        1, 0, 1, 1, 0,
        0, 1, 0, 0, 1,
        1, 1, 0, 0, 1
    ]
})

print("Исходный датасет:")
print(df)

#НОМЕР 1
X = df.drop(columns=["spam"])
Y = df["spam"]
#НОМЕР 2
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state = 42,
)
#НОМЕР 3
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
#НОМЕР 4
predictions = model.predict(X_test)
#НОМЕР 5
accuracy = accuracy_score(Y_test,predictions)
precision = precision_score(Y_test,predictions)
recall = recall_score(Y_test,predictions)
f1 = f1_score(Y_test,predictions)
print("Accuracy:", f'{accuracy:.3f}')
print("Precision:", f'{precision:.3f}')
print("Recall:", f'{recall:.3f}')
print("F1:", f'{f1:.3f}')
#НОМЕР 6
# Я думаю, что наиболее важной из всех метрик здесь будет f1 потому как, например, accuracy нельзя рассматривать
# отдельно от остальных метрик, а f1 как раз будет еще отражать насколько высоки 2 другие метрики(precision и recall). 
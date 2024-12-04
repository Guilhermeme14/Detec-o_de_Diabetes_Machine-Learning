import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("diabetes_pt.csv")

columns_with_zeros = [
    "Glicose",
    "PressaoArterial",
    "Espesura_da_Pele",
    "Insulina",
    "IMC",
]
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

data[columns_with_zeros] = data[columns_with_zeros].fillna(
    data[columns_with_zeros].mean()
)

X = data.drop(columns=["Resultado"])
y = data["Resultado"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Acurácia Regressão Logística: {accuracy_logistic * 100:.2f}%")
print(f"Acurácia Random Forest: {accuracy_rf * 100:.2f}%")
print(f"Acurácia SVM: {accuracy_svm * 100:.2f}%")


algorithms = ["Regressão Logística", "Random Forest", "SVM"]
accuracies = [accuracy_logistic * 100, accuracy_rf * 100, accuracy_svm * 100]

accuracies = [float(acc) for acc in accuracies]

plt.figure(figsize=(8, 6))
plt.bar(algorithms, accuracies, width=0.5)
plt.title("Acurácia dos Algoritmos de Machine Learning", fontsize=14)
plt.ylabel("Acurácia (%)", fontsize=12)
plt.xlabel("Algoritmos", fontsize=12)
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("algorithm_accuracies.png")
plt.close()

print("Gráfico de acurácia salvo como 'algorithm_accuracies.png'")

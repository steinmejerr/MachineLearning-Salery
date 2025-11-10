import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

CSV_PATH = "Salary_dataset.csv" 

def main():

    df = pd.read_csv(CSV_PATH)
    print(df.head())

    plt.figure()
    plt.scatter(df["YearsExperience"], df["Salary"], alpha=0.85)
    plt.title("YearsExperience vs. Salary")
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.tight_layout()

    X = df[["YearsExperience"]]
    y = df["Salary"]

    model = LinearRegression().fit(X, y)

    experience_years = 4.5
    pred = float(model.predict(pd.DataFrame({'YearsExperience':[experience_years]}))[0])
    print(f"Forventet løn for {experience_years} års erfaring: {pred:.2f}")

    x_min, x_max = df["YearsExperience"].min(), df["YearsExperience"].max()
    X_line = pd.DataFrame({"YearsExperience": np.linspace(x_min, x_max, 200)})
    y_line = model.predict(X_line)

    plt.figure()
    plt.scatter(df["YearsExperience"], df["Salary"], alpha=0.85, label="Data")
    plt.plot(X_line["YearsExperience"], y_line, label="Linear Regression")
    plt.title("Linear Regression Fit")
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.legend()
    plt.tight_layout()
    plt.show()

    dump(model, "salary_model.joblib")
    print("Model gemt")

if __name__ == "__main__":
    main()
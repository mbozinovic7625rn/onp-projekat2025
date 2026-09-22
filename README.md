# Air Quality Analysis — Fundamentals of Data Science (ONP) Project

End-to-end data science project on the **UCI Air Quality dataset**, built for the *Osnove nauke o podacima* (Fundamentals of Data Science) course at Računarski fakultet (RAF), Belgrade — 2025.

The project walks through the full data science workflow in seven phases, from raw sensor data to trained forecasting models.

## Project phases

1. **Dataset & description** — loading the UCI Air Quality dataset (hourly gas-sensor readings from an Italian city) and documenting its structure and quirks (`;` separators, decimal commas, `-200` missing-value sentinels).
2. **Visualization** — dynamic and static plots of pollutant concentrations and sensor behavior over time.
3. **Data preparation & EDA** — cleaning, correlation analysis, and imputation of missing values using a **KNN imputer**.
4. **Statistical analysis & hypothesis testing** — normality checks with **Shapiro–Wilk** and group comparisons with the **Mann–Whitney U** test.
5. **Predictive modeling** — baseline **linear regression** and **SVR**, including a comparison of different SVR kernels with **GridSearchCV** hyperparameter tuning.
6. **Advanced machine learning** — **K-Means** clustering with **PCA**, a **Decision Tree regressor**, a **neural-network regressor** (TensorFlow/Keras with early stopping), and a head-to-head comparison of the two.
7. **Time series** — stationarity testing (**ADF**), seasonal decomposition, **SARIMAX** forecasting (auto-ARIMA order selection via `pmdarima`), and an **LSTM** model.

## Repository contents

| File | Description |
|---|---|
| `mihailo_bozinovic_7625_rn.ipynb` | Main notebook — all seven phases with commentary and conclusions |
| `projekat.py` | Script version of the analysis |
| `mihailo_bozinovic_76_25_rn.pptx` | Defense presentation |
| `Projekat2025 - ONP.pdf` | Official project assignment |
| `teorija_za_odbranu.txt` | Theory notes prepared for the project defense |

## Tech stack

`Python` · `pandas` · `NumPy` · `matplotlib` / `seaborn` · `scikit-learn` · `SciPy` · `statsmodels` · `pmdarima` · `TensorFlow / Keras`

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels pmdarima tensorflow
jupyter notebook mihailo_bozinovic_7625_rn.ipynb
```

The notebook expects `AirQuality.csv` (UCI Air Quality dataset) in the working directory — available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality).

## Author

Mihailo Božinović — RN 76/2025, Računarski fakultet (RAF), Belgrade

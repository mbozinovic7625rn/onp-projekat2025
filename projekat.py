# %% [markdown]
# #### Uključivanje potrebnih biblioteka
# - Biblioteka pandas - omogućava nam rad sad dataset-om (mogućnost čitanja, pisanja, modifikacije i sl.)
# - Biblioteka numpy - omogućava nam rad sa matematičkim funkcijama, nizovima, listama i sl.
# - Bibilioteka matplotlib.pyplot - omogućava nam crtanje grafikona
# - Bibilioteka seaborn - omogućava nam detaljnije i naprednije crtanje grafika za analizu

# %%
import pandas as pd
import numpy as np

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import math

# %%
from scipy.stats import shapiro, mannwhitneyu, norm
import statistics

# %% [markdown]
# # Uključivanje dataset-a i njegov opis (Faza 1)
# - Link: https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set?resource=download&select=AirQuality.csv
# - Opis:
#     - Dataset sadrži podatke dobijene od senzora zagadjenosti vazduha postavljenog u Italijanskom gradu od 2004. do 2005. godine
#     - Kolone:
#         - Datum (DD/MM/YYYY)
#         - Vreme (HH.MM.SS)
#         - Tačna jednočasovna prosečna koncentracija CO u mg/m³
#         - PT08.S1 (kalaj-oksid) – prosečan jednočasovni odziv senzora (nominalno cilja CO)
#         - Tačna jednočasovna prosečna koncentracija nemetanskih ugljovodonika u µg/m³ (referentni analizator)
#         - Tačna jednočasovna prosečna koncentracija benzena u µg/m³ (referentni analizator)
#         - PT08.S2 (titanijum-dioksid) – prosečan jednočasovni odziv senzora (nominalno cilja NMHC)
#         - Tačna jednočasovna prosečna koncentracija NOx u ppb (referentni analizator)
#         - PT08.S3 (volfram-oksid) – prosečan jednočasovni odziv senzora (nominalno cilja NOx)
#         - Tačna jednočasovna prosečna koncentracija NO₂ u µg/m³ (referentni analizator)
#         - PT08.S4 (volfram-oksid) – prosečan jednočasovni odziv senzora (nominalno cilja NO₂)
#         - PT08.S5 (indijum-oksid) – prosečan jednočasovni odziv senzora (nominalno cilja O₃)
#         - Temperatura u °C
#         - Relativna vlažnost vazduha (%)
#         - AH – apsolutna vlažnost
#
# - Skup podataka sadrži 9357 instanci sa satnim prosečnim vrednostima odziva niza od 5 metal-oksidnih hemijskih senzora ugrađenih u uređaj za merenje kvaliteta vazduha. Uređaj je bio postavljen na terenu u značajno zagađenom području, na nivou puta, u jednom italijanskom gradu. Podaci su prikupljani od marta 2004. do februara 2005. godine (jedna godina), što predstavlja najduže slobodno dostupno snimanje odziva senzora za kvalitet vazduha postavljenih na terenu.
# - Referentne satne prosečne koncentracije za CO, nemetanske ugljovodonike, benzen, ukupne azotne okside (NOx) i azot-dioksid (NO₂) obezbeđene su od strane sertifikovanog analizatora koji je bio postavljen zajedno sa uređajem. U podacima postoje dokazi o unakrsnoj osetljivosti, kao i o „concept drift“ i „sensor drift“ pojavama, što je opisano u radu De Vito i saradnici, Sens. And Act. B, Vol. 129,2,2008 (navod je obavezan), a što eventualno utiče na sposobnosti senzora da precizno procene koncentracije.
# - Nedostajuće vrednosti su označene sa vrednošću -200.

# %%
df = pd.read_csv(
    "AirQuality.csv", sep=";"
)  # koristimo sep=';' jer su kolone odvojene sa tačka-zarezom

# %% [markdown]
# Koristeći funkciju .head(int n) možemo dobiti prvih n redova za dataset-a sa nazivima kolona, kako bi mogli da vidimo primer šta se nalazi u istom.

# %%
df.head(10)

# %% [markdown]
# Koristeći funkciju .shape, možemo dobiti dimenzije našeg dataset-a (u našem slučaju se one nalaze ispod 9471 red i 17 kolona)

# %%
df.shape

# %% [markdown]
# Koristeći .describe() funkciju možemo da dobijemo neke osnovne statističke podatke vezane za naš dataset

# %%
df.describe()

# %% [markdown]
# Koristeći funkciju .info(), možemo dobiti podatke o tipovima kolona, da li je broj ili objekat (sve ostalo)

# %%
df.info()

# %% [markdown]
# # Vizuelizacija podataka i dinamičko predstavljanje (Faza 2)

# %% [markdown]
# Za početak prikazujemo distribuciju parametra koji imaju float64 vrednosti, da bi prikazali grafikon kako treba, sklonio sam NaN i -200 vrednosti.
# U suprotnom bi nam grafik izgledao dosta drugačije.

# %%
float64_cols = df.select_dtypes(include=["float64"]).columns

# %%
n = len(float64_cols)
cols = 4
rows = math.ceil(n / cols)

plt.figure(figsize=(5 * cols, 4 * rows))

# Plot each column
for i, col in enumerate(float64_cols):
    plt.subplot(rows, cols, i + 1)
    sns.histplot(
        df[col][(df[col] != -200) & (df[col].notna())], kde=True, color="skyblue"
    )
    plt.title(f"Distribucija {col}")
    plt.xlabel(col)
    plt.ylabel("Frekvencija")
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# Iz prethodnog grid-a možemo da zaključimo da su sve naše kolone koje ne zahtevaju neku obradu podataka poprilično normalno distribuirane. Vidimo da nema nekih odstupanja tu.
# Sada prikazujemo tzv. BoxPlot. On nam govori koliko "outliers"-a imaju naše kolone u datasetu. Vidimo da svaka od naših kolona ima dosta outliers-a. Tako da ćemo u EDA delu da probamo da smanjimo taj broj na minimum. Takodje sleduje nam u tom delu i čišćenje NaN vrednosti i konverzija -200 u NaN takodje.

# %%
cols = 4
rows = math.ceil(len(float64_cols) / cols)

plt.figure(figsize=(6 * cols, 4 * rows))

for i, col in enumerate(float64_cols):
    # Clean the column
    clean_col = df[col].replace(-200, pd.NA).dropna()

    # Skip if empty after cleaning
    if clean_col.empty:
        continue

    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=clean_col, color="lightcoral")
    plt.title(f"Boxplot: {col}")
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# U sledećem box-u možemo da prikažemo promenu naših promenljivih kroz vreme, tj. po indeksima jer još uvek nismo spojili naše kolone u datetime kolonu (to ćemo uraditi u EDA delu). Ovako možemo pratiti kako otprilike izgleda promena naših numeričkih kolona.

# %%
rows = math.ceil(len(float64_cols) / cols)
plt.figure(figsize=(6 * cols, 4 * rows))

for i, col in enumerate(float64_cols):
    # Clean the column: replace -200 with NaN and drop missing values
    clean_col = df[col].replace(-200, np.nan).dropna()
    if clean_col.empty:
        continue

    plt.subplot(rows, cols, i + 1)
    plt.plot(clean_col.values, color="royalblue")
    plt.title(f"Time Series: {col}")
    plt.xlabel("Vreme (indeks)")
    plt.ylabel(col)
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# Vidimo da je bilo nekih odskakanja u PT08.S3(NOx), i da je recimo NMHC(GT) jako drugačija kroz vreme, tj. da ima odskakanja. dok su sve ostale poprilično stabilne osim recimo NOx(GT) jer ona je imala malo blaži početak i u nekom trenutku je krenula da raste.

# %% [markdown]
# Sada ćemo napraviti scatter plot, on je sličan našem plotu koji ide kroz vreme. Možda ima malo detaljinij prikaz kada se odstupanja jasnije vide.

# %%
rows = math.ceil(len(float64_cols) / cols)
plt.figure(figsize=(6 * cols, 4 * rows))

for i, col in enumerate(float64_cols):
    clean_col = df[col].replace(-200, np.nan).dropna()
    if clean_col.empty:
        continue

    plt.subplot(rows, cols, i + 1)
    plt.scatter(range(len(clean_col)), clean_col, c="mediumseagreen", alpha=0.6, s=10)
    plt.title(f"Scatter plot: {col} kroz vreme (indeks)")
    plt.xlabel("Vreme (indeks)")
    plt.ylabel(col)
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# Sada možemo naslikati matricu korelacije. Korelacija služi da mi vidimo u kakvom su odnosu naše kolone, tj. možemo da vidimo šta od čega zavisi u našem dataset-u.

# %%
sns.heatmap(
    df[float64_cols].replace(-200, np.nan).corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
)
plt.title("Matrica korelacije naših numeričkih kolona!")
plt.show()

# %% [markdown]
# Odavde možemo da zaključimo da su naše kolone Unnamed: 15 i 16, nepotrebne, te ćemo iste obrisati kasnije u EDA delu.
# Takodje raspodela je sledeća:
# 1) Ako je korelacija veća od 0.5, onda kažemo da su u jakoj pozitivnoj korelaciji
# 2) Ako je izmedju 0.3 i 0.49, onda kažemo da su u okej pozitivnoj korelaciji
# 3) AKo je izmedju 0.1 i 0.29, onda su u baš slaboj pozitivnoj korelaciji.
# 4) Isto važi i za negativne, samo su oni u negativnoj korelaciji.
#
# Razlika izmedju negativne i pozitivne korelacije:
# - U pozitivnoj korelaciji što su kolone u većoj korelaciji to zavise više jedna od druge tj. kako jedna raste i druga raste ili kako jedna opada i druga opada
# - U negativnoj je totalno suprotno, kako jedna raste tako druga opada ili kako jedna opada druga raste.
#

# %% [markdown]
# # Priprema podatak i istraživačka analiza (Faza 3)

# %% [markdown]
# - Prva stavka koju ćemo odraditi je provera nedostajućih vrednosti, konverzija podataka koji nisu dobri i čišćenje nepotrebnih kolona koje smo videli u prethodnom delu.

# %%
df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"])

# %%
df.head(5)

# %% [markdown]
# - Konkretno u ovom koraku smo obrisali kolone 'Unnamed: 15' i 'Unnamed: 16' jer su nepotrebno, tj. sve su NaN vrednosti
# - Sledeći korak će biti konverzija podataka koji su object: CO(GT), C6H6(GT), T, RH, AH - ove kolone imaju , umesto . kao decimalni razmak

# %%
cols = [
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "T",
    "RH",
    "AH",
]  # kolone koje su object tipa, a treba da budu float
df[cols] = df[cols].astype(str).replace(",", ".", regex=True).astype(float)

# %%
df.info()

# %% [markdown]
# - Sada vidimo da smo pravilno konvertovali kolone u odgovarajući float64 format.
# - Sledeći korak je da pretvorimo kolone Date i Time u Datetime format.

# %%
df["DateTime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S"
)
df.insert(0, "DateTime", df.pop("DateTime"))  # premeštamo DateTime na početak

# %%
df.head(2)

# %% [markdown]
# - Sledeći korak može biti uklanjanje Date i Time kolona.
# - Takodje ćemo da zamenimo -200 vrednosti sa NaN vrednostima, kako bi lakše manipulisali istim.

# %%
df = df.drop(columns=["Date", "Time"])

# %%
df = df.replace(-200, np.nan)

# %%
(df.isna().sum() / len(df)) * 100  # prikazujemo procenat NaN vrednosti po kolonama

# %% [markdown]
# ### Zaključci:
# - Vidimo da sve vrednosti osim NMHC(GT) imaju normalan broj nedostajućih vrednosti
# - Kolona NMHC(GT) ima 90.3% nedostajućih vrednosti, što znači da možemo da je obrišemo
# - Ako bi je eventualno imputovali, to ne bi bilo dobro po kasnije stavke ovog projekta, tj. treniranje modela

# %%
df = df.drop(columns=["NMHC(GT)"])

# %%
df.head(2)

# %% [markdown]
# - Takodje s obzirom da nam DateTime kolona ima NaN vrednosti, i da nju nije moguće Impute-ovati (dopuniti pomoću nekog od Imputer-a nedostajuće vrednosti), najbolje bi bilo da obrišemo te kolone
# - Svakako je procenat istih ~1.2% tako da neće mnogo uticati na naš dataset

# %%
df = df.dropna(subset=["DateTime"])

# %% [markdown]
# - Hajde da proverimo ima li duplikata u našem dataset-u

# %%
(df.duplicated() == True).sum()

# %% [markdown]
# - Vidimo da nemamo duplikate u našem dataset-u

# %% [markdown]
# ### Korelacije
# - Hajde da ponovo vidimo heatmap, sada kada smo pročistili naš dataset.

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrica korelacije naših kolona", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# - Sada možemo precizno videti korelacije u našem dataset-u
# - Pravila su ista kao i prethodno opisana u gornjem delu notebook-a

# %% [markdown]
# ### Uklanjanje nedostajućih NaN vrednosti
# - Za ovo će nam biti potreban Imputer
# - Za kolone koje imaju mali broj NaN do ~6% ćemo koristiti SimpleImputer
# - Za kolone koje imaju veći broj NaN vrednosti do ~20% ćemo korisiti KNNImputer
# --------------------------
# - CO(GT)           18.973709
# - PT08.S1(CO)       5.068103
# - C6H6(GT)          5.068103
# - PT08.S2(NMHC)     5.068103
# - NOx(GT)          18.509133
# - PT08.S3(NOx)      5.068103
# - NO2(GT)          18.540809
# - PT08.S4(NO2)      5.068103
# - PT08.S5(O3)       5.068103
# - T                 5.068103
# - RH                5.068103
# - AH                5.068103

# %%
from sklearn.impute import KNNImputer, SimpleImputer

# %%
knn_feautres = ["CO(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]
simple_features = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "C6H6(GT)",
]

# %%
simpleImputer = SimpleImputer(strategy="mean")
knnImputer = KNNImputer(
    n_neighbors=8
)  # ovo smo postavili na 8, kako bi imali pristup većem broju susednih vrednosti

# %%
df[knn_feautres] = knnImputer.fit_transform(df[knn_feautres])
df[simple_features] = simpleImputer.fit_transform(df[simple_features])

# %%
df.isna().sum()

# %% [markdown]
# #### Zaključci:
# - Imputovanjem smo se otarasili NaN vrednosti koje bi nam smetale u kasnijem procesiranju
# - Za kolone koje imaju ~18% nedostajućih vrednosti smo koristili KNNImputer, kako bi imali pristup nekom broju susednih vrednost i kako bi što preciznije popunili te podatke
# - Za kolone koje su bile na ~5% smo koristili SimpleImputer, sa strategijom mean koja funkcioniše tako što na osnovu mean-a popunjava nedostajuće vrednosti u našim kolonama koje smo izabrali
# - Takodje će nam usled velikih razlika u vrednostima biti potreban Scaler, kako bi mogli da ujednačimo naše vrednosti, tj. opseg naših vrednosti jer npr:
#     - CO(GT) ima opseg od 0-10
#     - PT08.S1(CO) ima opseg od 500-2000
#     - ...
# - Tako da bi smo mogli da koristimo ili StandardScaler ili MinMaxScaler
# - Mi ćemo ovde koristiti StandardScaler, zato što iz faze 2 mi možemo da zaključimo da naše kolone imaju normalnu distribuciju
# - Takodje ćemo sačuvati DateTime kolonu kako bi je koristili kasnije za vremenske serije (time series)

# %% [markdown]
# - Dodatak na ovo, pre skaliranja, hajde da obrišemo sve outlier-e iz našeg dataset-a
# - Koristićemo IQR kako bi uspešno pronašli outlier-e i sklonili ih iz dataset-a

# %%
# IQR metod za uklanjanje outliera
numeric_cols = df.select_dtypes(include=[np.number]).columns

before_count = len(df)
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Ignorišemo IQR == 0
lower = lower.where(IQR != 0, -np.inf)
upper = upper.where(IQR != 0, np.inf)

mask = df[numeric_cols].ge(lower) & df[numeric_cols].le(upper)
mask_all = mask.all(axis=1)

df = df[mask_all].reset_index(drop=True)

after_count = len(df)
print(
    f"Obrisane {before_count - after_count} kolona ({(before_count - after_count)/before_count:.2%}) - nove dimenzije: {df.shape}"
)

# %%
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
dt_col = df["DateTime"]
df = df.drop(columns=["DateTime"])
temperature_original = df["T"].copy()  # kopirana T zbog naše vremenske serije :)
df[df.columns] = standardScaler.fit_transform(df[df.columns])
df.insert(0, "DateTime", dt_col)

# %%
df.head(5)

# %% [markdown]
# # Statistička analiza i testiranje hipoteza (Faza 4)
# - Uzećemo naše skalirane kolone koje imaju numeričke vrednosti kako bi radili ispitivanja na njima
# - Kolone su sledeće:
#     - T,
#     - RH,
#     - AH.

# %%
custom_cols = ["T", "RH", "AH"]

# %%
n = len(custom_cols)
cols = 2
rows = math.ceil(n / cols)

plt.figure(figsize=(5 * cols, 4 * rows))

# Plot each column
for i, col in enumerate(custom_cols):
    plt.subplot(rows, cols, i + 1)
    sns.histplot(df[col], kde=True, color="red")
    plt.title(f"Distribucija {col}")
    plt.xlabel(col)
    plt.ylabel("Frekvencija")
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# - H0 za naše kolone: Podaci imaju normalnu distribuciju
# - H1 za naše kolone: Podaci nemaju normalnu distribuciju.

# %%
# Provera normalnosti pomoću Shapiro-Wilk testa
for col in custom_cols:
    p_value = shapiro(df[col]).pvalue
    print(f"Shapiro-Wilk test za kolonu {col}: p={p_value:.4f}")
    if p_value > 0.05:
        print(
            f"Kolona {col} verovatno prati normalnu distribuciju (ne odbacujemo H0)\n"
        )
    else:
        print(
            f"Kolona {col} verovatno ne prati normalnu distribuciju (odbacujemo H0)\n"
        )

# %% [markdown]
# S' obzirom da nam Shapiro-Wilk test, daje rezultate da niko ne prati normalnu distribuciju(što se i vidi sa našeg histoplot-a), iskoristićemo Mann-Whitney test izmedju nekih od kolona (T i AH, T i RH), kako bi videli da li bi zaista trebalo da odbacimo H0.
#
# - H0 za T i AH - Nema velike razlike u srednjoj vrednosti temperature i apsolutne vlažnosti.
# - H1 za T i AH - Ima velike razlike u srednjoj vrednosti.
# - H0 za T i RH - Distribucija temperature i relativne vlažnosti su iste,
# - H1 za T i RH - Značajno se razlikuju.

# %%
u_stat, p_value = mannwhitneyu(df["T"], df["AH"], alternative="two-sided")
print(f"Mann-Whitney U test između T i AH: U={u_stat}, p={p_value:.4f}")
if p_value < 0.05:
    print("Postoji značajna razlika između distribucija T i AH (odbacujemo H0)")
else:
    print("Nema značajne razlike između distribucija T i AH (ne odbacujemo H0)")

# %%
u_stat, p_value = mannwhitneyu(df["T"], df["RH"], alternative="two-sided")
print(f"Mann-Whitney U test između T i RH: U={u_stat}, p={p_value:.4f}")
if p_value < 0.05:
    print("Postoji značajna razlika između distribucija T i RH (odbacujemo H0)")
else:
    print("Nema značajne razlike između distribucija T i RH (ne odbacujemo H0)")

# %% [markdown]
# Koristimo Cohen-ov d, kako bi odredili kolika je razlika izmedju skupova T, RH i AH. Uzeli smo njih kao primere, možemo uzeti i ostale skupove kako bi imali veći prikaz.

# %%
std = math.sqrt(
    (
        (len(df["T"]) - 1) * df["T"].std() ** 2
        + (len(df["AH"]) - 1) * df["AH"].std() ** 2
    )
    / (len(df["T"]) + len(df["AH"]) - 2)
)

cohens_d = statistics.mean(df["T"]) - statistics.mean(df["AH"]) / std
print(f"Cohen-ov d između T i AH: {cohens_d:.4f}")

# Tumačenje efekta
if abs(cohens_d) < 0.2:
    print("Veličina efekta je mala.")
elif abs(cohens_d) < 0.5:
    print("Veličina efekta je srednja.")
else:
    print("Veličina efekta je velika.")

# %%
std = math.sqrt(
    (
        (len(df["T"]) - 1) * df["T"].std() ** 2
        + (len(df["RH"]) - 1) * df["RH"].std() ** 2
    )
    / (len(df["T"]) + len(df["RH"]) - 2)
)

cohens_d = statistics.mean(df["T"]) - statistics.mean(df["RH"]) / std
print(f"Cohen-ov d između T i RH: {cohens_d:.4f}")

# Tumačenje efekta
if abs(cohens_d) < 0.2:
    print("Veličina efekta je mala.")
elif abs(cohens_d) < 0.5:
    print("Veličina efekta je srednja.")
else:
    print("Veličina efekta je velika.")

# %%
std = math.sqrt(
    (
        (len(df["AH"]) - 1) * df["AH"].std() ** 2
        + (len(df["RH"]) - 1) * df["RH"].std() ** 2
    )
    / (len(df["AH"]) + len(df["RH"]) - 2)
)

cohens_d = statistics.mean(df["AH"]) - statistics.mean(df["RH"]) / std
print(f"Cohen-ov d između AH i RH: {cohens_d:.4f}")

# Tumačenje efekta
if abs(cohens_d) < 0.2:
    print("Veličina efekta je mala.")
elif abs(cohens_d) < 0.5:
    print("Veličina efekta je srednja.")
else:
    print("Veličina efekta je velika.")

# %% [markdown]
# Intervali poverenja - služe kako bi dali opseg u kojem očekujemo da se nadje srednja vrednost kolone, sa odredjenim nivoom sigurnosti

# %%
data = df["T"]

# Parametri uzorka
mean_val = data.mean()
std_dev = data.std(ddof=1)  # standardna devijacija uzorka
n = len(data)

# 95% interval poverenja
z_value = norm.ppf(0.975)  # 1.96
margin_error = z_value * (std_dev / math.sqrt(n))
conf_interval = (mean_val - margin_error, mean_val + margin_error)

print(f"Prosečna vrednost (T): {mean_val:.4f}")
print(f"95% interval poverenja: {conf_interval}")

# %%
data = df["RH"]

# Parametri uzorka
mean_val = data.mean()
std_dev = data.std(ddof=1)  # standardna devijacija uzorka
n = len(data)

# 95% interval poverenja
z_value = norm.ppf(0.975)  # 1.96
margin_error = z_value * (std_dev / math.sqrt(n))
conf_interval = (mean_val - margin_error, mean_val + margin_error)

print(f"Prosečna vrednost (RH): {mean_val:.4f}")
print(f"95% interval poverenja: {conf_interval}")

# %%
data = df["AH"]

# Parametri uzorka
mean_val = data.mean()
std_dev = data.std(ddof=1)  # standardna devijacija uzorka
n = len(data)

# 95% interval poverenja
z_value = norm.ppf(0.975)  # 1.96
margin_error = z_value * (std_dev / math.sqrt(n))
conf_interval = (mean_val - margin_error, mean_val + margin_error)

print(f"Prosečna vrednost (AH): {mean_val:.4f}")
print(f"95% interval poverenja: {conf_interval}")

# %% [markdown]
# Zaključak:
# - S' obzirom da smo uradili skaliranje naših podataka, to znači da smo ih ili spustili na interval [-1, 1] dobijamo da je prosečna vrednost 0, što i jeste tačno za date kolone :D
# - Takodje, vidimo da je interval poverenja u skaliranoj verziji od -0.02 do 0.02 (približno, za sve tri kolone se razlikuju)
# - Zato ne možemo da imamo precizne prosečne vrednosti

# %% [markdown]
# # Prediktivno modeliranje i mašinsko učenje (Faza 5)
# - U našem slučaju radićemo sa regresijom, pošto nemamo nikakvu klasifikaciju u našem izabranom dataset-u
# - Target varijabla za predikciju biće u našem slučaju  T - da vidimo kako bi naš model predvideo temperaturu

# %% [markdown]
# Ovde radimo odvajanje podataka, tj. postavljamo našu target varijablu i naše feature. Kako bi posle mogli da primenimo train_test_split.

# %%
target = df["T"]
features = df.drop(columns=["T", "DateTime"])

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
print(f"Dimenzije X_train: {X_train.shape}")
print(f"Dimenzije X_test: {X_test.shape}")

# %%
from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV,
)  # Linearna, Ridge i Lasso regresija
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)  # metrike za evaluaciju regresionih modela
from sklearn.svm import SVR  # Support Vector Regression

# %%
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred_lin_reg = linear_regression.predict(X_test)

lin_reg_r2 = r2_score(y_test, y_pred_lin_reg)
lin_reg_mae = mean_absolute_error(y_test, y_pred_lin_reg)
lin_reg_mse = mean_squared_error(y_test, y_pred_lin_reg)
lin_reg_rmse = root_mean_squared_error(y_test, y_pred_lin_reg)

print(f"Linearna regresija, R^2: {lin_reg_r2:.4f}")
print(f"Linearna regresija,  MAE: {lin_reg_mae:.4f}")
print(f"Linearna regresija, MSE: {lin_reg_mse:.4f}")
print(f"Linearna regresija, RMSE: {lin_reg_rmse:.4f}")

# %% [markdown]
# Ovde možemo videti naše R^2, MAE, MSE i RMSE:
# - R^2:
#     - Mera koliko dobro naš model objašnjava varijansu ciljne promenljve
#     - Kreće se od 0 do 1 (može biti negativan kod loših modela)
#     - U našem slučaju je 0.9333, što znači da naš model objašnjava oko 93.33% varijacije u podacima
#     - To je vrlo dobro
# - MAE (Mean absolute error):
#     - Proesečna apsolutna greška predikcije
#     - Računa se kao prosečna vrednost apsolutne vrednosti |y_pred - y_i| i = 1, 2, 3, ..., n
#     - Što je MAE niži, to je naš model bolji, tj. vrednosti predikcija su bliže pravim vrednostima
#     - U našem slučaju MAE = 0.1932, što znači da model u proseku greši za 0.19 jedinica
# - MSE (Mean squared error):
#     - Prosečna kvadratna greška predikcije
#     - Umesto apsolutne vrednosti imamo, kvadrat u formuli za izračunavanje
#     - Isto važi kao i za MAE, što niže to bolje
#     - U našem slučaju, MSE = 0.0682, što znači da model ima male greške, s obzirom da su greške kvadratne
#     - Jače je od MAE u smislu provere
# - RMSE (Root mean squared error):
#     - Dobija se kao koren od MSE (sqrt(MSE))
#     - Vraća grešku i istim jedinicama kao ciljna promenljiva
#     - Kao i MSE, kažnjava veće greške "jače" nego MAE zbog kvadriranja
#     - U našem slučaju RMSE = 0.2611, što znači da model u proseku greši za oko 0.26 jedinica

# %% [markdown]
# Kod Ridge i Lasso regresije, parametar alfa se koristi za podešavanje "snage" regularizacije.
# U ovom slučaju koristim Cross-Validation, kako bih našao najbolji parametar alfa i za Ridge i za Lasso.

# %%
# Testiramo više alpha vrednosti automatski
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="r2")
ridge_cv.fit(X_train, y_train)

print(f"Najbolji alpha: {ridge_cv.alpha_}")

y_pred_ridge = ridge_cv.predict(X_test)

ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)

print(f"Ridge regresija, R^2: {ridge_r2:.4f}")
print(f"Ridge regresija, MAE: {ridge_mae:.4f}")
print(f"Ridge regresija, MSE: {ridge_mse:.4f}")
print(f"Ridge regresija, RMSE: {ridge_rmse:.4f}")

# %%
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)

print(f"Najbolji alpha: {lasso_cv.alpha_}")

y_pred_lasso = lasso_cv.predict(X_test)

lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(lasso_mse)

print(f"Lasso regresija, R^2: {lasso_r2:.4f}")
print(f"Lasso regresija, MAE: {lasso_mae:.4f}")
print(f"Lasso regresija, MSE: {lasso_mse:.4f}")
print(f"Lasso regresija, RMSE: {lasso_rmse:.4f}")

# %% [markdown]
# | Model                | R²      | MAE     | MSE     | RMSE    | Najbolji α |
# |---------------------|---------|---------|---------|---------|------------|
# | Linearna regresija  | 0.9333  | 0.1932  | 0.0682  | 0.2612  | -          |
# | Lasso regresija     | 0.9333  | 0.1923  | 0.0682  | 0.2611  | 0.001      |
# | Ridge regresija     | 0.9333  | 0.1932  | 0.0682  | 0.2611  | 1.0        |

# %% [markdown]
# Kako nam je alfa = 1 u Ridge regresiji, to se svodi na Linearnu regresiju
# Kako nam je alfa = 0.001 za Lasso regresiju, to znači da Lasso ima minimalnu regularizaciju i da je skoro isto kao i obična linearna regresija
# Uglavnom, ovde možemo koristiti i samo Linearnu regresiju.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Linear Regression
axes[0].scatter(y_test, y_pred_lin_reg, alpha=0.5, color="blue")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0].set_xlabel("Stvarne vrednosti", fontsize=12)
axes[0].set_ylabel("Predviđene vrednosti", fontsize=12)
axes[0].set_title(f"Linearna regresija\nR² = {lin_reg_r2:.4f}", fontsize=14)
axes[0].grid(True, alpha=0.3)

# Ridge Regression
axes[1].scatter(y_test, y_pred_ridge, alpha=0.5, color="green")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[1].set_xlabel("Stvarne vrednosti", fontsize=12)
axes[1].set_ylabel("Predviđene vrednosti", fontsize=12)
axes[1].set_title(f"Ridge regresija\nR² = {ridge_r2:.4f}", fontsize=14)
axes[1].grid(True, alpha=0.3)

# Lasso Regression
axes[2].scatter(y_test, y_pred_lasso, alpha=0.5, color="orange")
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[2].set_xlabel("Stvarne vrednosti", fontsize=12)
axes[2].set_ylabel("Predviđene vrednosti", fontsize=12)
axes[2].set_title(f"Lasso regresija\nR² = {lasso_r2:.4f}", fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# Ovde ću da probam SVM (Support Vector Machine), tj. u našem slučaju SVR (Support Vector Regression)

# %% [markdown]
# ### Testiranje različitih SVR kernela

# %%
kernels = ["linear", "rbf", "poly", "sigmoid"]
svr_results = {}

for kernel in kernels:
    print(f"Treniranje SVR sa {kernel} kernelom...")
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    svr_results[kernel] = {
        "model": svr,
        "y_pred": y_pred,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }

    print(f"R^2 = {r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}\n")

best_kernel = max(svr_results, key=lambda k: svr_results[k]["R2"])
print(f"Najbolji kernel: {best_kernel.upper()}")
print(f"R^2 = {svr_results[best_kernel]['R2']:.4f}")

# %%
# Vizualizacija poređenja metrika za različite kernele
metrics = ["R2", "MAE", "MSE", "RMSE"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    values = [svr_results[k][metric] for k in kernels]
    colors = ["green" if k == best_kernel else "lightblue" for k in kernels]

    axes[idx].bar(kernels, values, color=colors, alpha=0.7, edgecolor="black")
    axes[idx].set_title(
        f"Poređenje {metric} za različite kernele", fontsize=12, fontweight="bold"
    )
    axes[idx].set_xlabel("Kernel", fontsize=11)
    axes[idx].set_ylabel(metric, fontsize=11)
    axes[idx].grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(values):
        axes[idx].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# Zaključak:
# - Poredili smo Lasso, Ridge i Linearnu regresiju, i videli da nam je od ta tri, zapravo sve jedno šta koristimo.
# - Sa najboljim alfa parametrom u oba slučaja dobijamo "aproksimaciju" linearne regresije.
# - Znamo da su nam najbolje metrike na linearnoj regresiji sledeće:
#     - R^2: 0.9333
#     - MAE: 0.1932
#     - MSE: 0.0682
#     - RMSE: 0.2611
# - Ali posle treniranja SVR-a, vidimo da nam je ipak najbolje da koristimo SVR, sa kernelom RBF.
# - Metrike za naš SVR su sledeće:
#     - R^2: 0.9959
#     - MAE: 0.0495
#     - MSE: 0.0042
#     - RMSE: 0.0648
# - Vidimo da nam naš SVR model ima mnogo bolje R^2 (0.9959 vs 0.9333), što znači da objašnjava 99.59% varijanse u podacima, u poređenju sa 93.33% kod linearne regresije.
# - Takođe, sve metrike grešaka (MAE, MSE, RMSE) su drastično smanjene kod SVR modela:
#     - MAE je smanjena sa 0.1932 na 0.0495
#     - MSE je smanjena sa 0.0682 na 0.0042
#     - RMSE je smanjena sa 0.2611 na 0.0648
# - Ovo nam govori da SVR model sa RBF kernelom može bolje da uhvati nelinearne odnose u podacima koje linearni modeli ne mogu da modeliraju.
# - Cross-validation rezultati potvrđuju stabilnost i generalizaciju našeg SVR modela.
# - Zaključujemo da je SVR sa RBF kernelom optimalan izbor za ovaj problem regresije, sa značajno boljim performansama od linearnih modela.

# %% [markdown]
# # Napredno mašinsko učenje (Faza 6)
# - Opet kao i u fazi 5, mi ćemo koristiti regresione modele
# - To je zato što nemamo nikakvu klasifikaciju u našem skupu podataka
# - Klasifikaciju smo mogli možda da dobijemo tako što bi neki opseg temperatura pretvarali u neke tekstualne podatke, npr. neka raspodela temperatura (toplo vreme, hladno vreme, i verovatno neki malo stručniji termini), ali svakako u našem slučaju ćemo koristiti regresione modele

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# %% [markdown]
# - U našem primeru, koristićemo ova 3 modela, dok ćemo naš neural network, pomoću "Backpropagation"-a, pokušati da svedemo na što bolji rezultat (tj. output)
# - Takodje ćemo imati i jedan model sa npr. 20 neurona, kako bi ispunili zahtev po specifikaciji
# - U "Backpropagation"-u, mi računamo gradijent loss funkcije, i prenosimo grešku nazad kroz nivoe našeg neural network-a, kako bi uspeli da optimizujemo naše parametre (težinu i bias), samim tim smanjili bi razliku izmedju predicted vrednosti i pravog izlaza koji mi očekujemo

# %% [markdown]
# ## K-Means
# - Na osnovu Elbow metode odredjujemo broj klastera za naš model
# - Crtamo grafikon i tamo gde se lomi kriva, dobijamo broj klastera

# %%
inertia_values = []
K_range = range(2, 11)  # Testiramo od 2 do 10 klastera

for k in K_range:
    # n_clusters: Broj klastera koji želimo da kreiramo
    # random_state: Seed za reproduktivnost rezultata
    # n_init: Broj puta da se algoritam pokreće sa različitim početnim centroidima (uzima se najbolji rezultat)
    # max_iter: Maksimalan broj iteracija za jedan run algoritma
    kmeans = KMeans(
        n_clusters=k,  # Broj klastera
        random_state=42,  # Za reproduktivnost
        n_init=10,  # Broj različitih inicijalizacija
        max_iter=300,  # Maksimalan broj iteracija
    )
    kmeans.fit(features)
    inertia_values.append(
        kmeans.inertia_
    )  # inertia_ je suma kvadratnih rastojanja do najbližeg centroida

# Vizualizacija Elbow metode
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia_values, "bo-", linewidth=2, markersize=8)
plt.xlabel("Broj klastera (K)", fontsize=12)
plt.ylabel("Inercija (suma kvadratnih rastojanja)", fontsize=12)
plt.title(
    "Elbow metoda za određivanje optimalnog broja klastera",
    fontsize=14,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.show()

# %% [markdown]
# - Na osnovu našeg grafikona, vidimo da bi mogli da uzmemo K = 4 ili K = 5, jer tu je najveći prelom na ovoj krivoj.
# - Probaćemo KMeans sa K = 4

# %%
kMeans4 = KMeans(
    n_clusters=4,  # Broj klastera
    random_state=42,  # Za reproduktivnost
    n_init=10,  # Broj različitih inicijalizacija
    max_iter=300,  # Maksimalan broj iteracija
)

# %%
kMeans4.fit(features)

# %%
cluster_labels_4 = (
    kMeans4.labels_
)  # labels_ sadrži klaster broj za svaki podatak (0, 1, 2, ...)

# %%
unique4, counts4 = np.unique(cluster_labels_4, return_counts=True)
print(f"Broj klastera: 4")
print("\nRaspodela podataka po klasterima:")
for cluster_id, count in zip(unique4, counts4):
    print(
        f"  Klaster {cluster_id}: {count} podataka ({count/len(cluster_labels_4)*100:.2f}%)"
    )

# Centroidi klastera
print(f"\nDimenzije centroida: {kMeans4.cluster_centers_.shape}")
print("Centroidi predstavljaju 'prosečnu' vrednost svih tačaka u klasteru")

# %%
df["Cluster_4"] = cluster_labels_4

# %%
print(df.groupby("Cluster_4")["T"].describe())

# %% [markdown]
# - Ove klastere bi mogli da interpetiramo na sledeći način:
#     1) Klaster 0: Hladno vreme (ujedno je i najveći klaster za hladnoću)
#     2) Klaster 1: Najtoplije vreme (ujedno i najveći topli klaster)
#     3) Klaster 2: Najhladnije vreme
#     4) Klaster 3: Umereno toplo vreme
# - Ova podela bazirana je na mean-u, tj. srednjoj vrednosti:
#     - Npr. klasteri su rasoporedjeni na sledeći način:
#         - klaster 2, klaster 0, klaster 3, klaster 1
#         - -0.83, -0.34, 0.039, 0.65
#     - Na osnovu toga smo doneli te zaključke

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels_4, s=30)
plt.title("Rezultati klasterovanja (K = 4) prikazani pomoću PCA")
plt.xlabel("Standardizovana karakteristika 1")
plt.ylabel("Standardizovana karakteristika 2")
plt.colorbar(label="Cluster")
plt.show()

# %%
kMeans5 = KMeans(
    n_clusters=5,  # Broj klastera
    random_state=42,  # Za reproduktivnost
    n_init=10,  # Broj različitih inicijalizacija
    max_iter=300,  # Maksimalan broj iteracija
)

# %%
kMeans5.fit(features)

# %%
cluster_labels_5 = (
    kMeans5.labels_
)  # labels_ sadrži klaster broj za svaki podatak (0, 1, 2, ...)

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels_5, s=30)
plt.title("Rezultati klasterovanja (K = 5) prikazani pomoću PCA")
plt.xlabel("Standardizovana karakteristika 1")
plt.ylabel("Standardizovana karakteristika 2")
plt.colorbar(label="Cluster")
plt.show()

# %%
kMeans3 = KMeans(
    n_clusters=3,  # Broj klastera
    random_state=42,  # Za reproduktivnost
    n_init=10,  # Broj različitih inicijalizacija
    max_iter=300,  # Maksimalan broj iteracija
)

# %%
kMeans3.fit(features)

# %%
cluster_labels_3 = (
    kMeans3.labels_
)  # labels_ sadrži klaster broj za svaki podatak (0, 1, 2, ...)

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels_3, s=30)
plt.title("Rezultati klasterovanja (K = 3) prikazani pomoću PCA")
plt.xlabel("Standardizovana karakteristika 1")
plt.ylabel("Standardizovana karakteristika 2")
plt.colorbar(label="Cluster")
plt.show()

# %% [markdown]
# ## Decision Tree Regressor
# - Svaki čvor u našem DTR predstavlja test nad jednom osobinom
# - Svaka grana predstavlja rezultat tog testa
# - Svaki list predstavlja predikciju (u našem slučaju pošto koristimo regresiju predstavlja kontinualnu vrednost)
# - Parametri su:
#     - max_depth - najveća dozovoljena dubina stabla
#     - criterion - kriterjum za greške MAE, MSE
#     - min_samples_split - minimalan broj uzoraka potreban da bi se čvor podelio
#     - min_samples_leaf - minimalan broj uzoraka u listu

# %%
dtr = DecisionTreeRegressor(
    random_state=42,  # Za reproduktivnost
    max_depth=None,  # Bez ograničenja dubine (može dovesti do overfitting-a)
    min_samples_split=2,  # Minimalno 2 uzorka za podelu čvora (default)
    min_samples_leaf=1,  # Minimalno 1 uzorak u listu (default)
    criterion="squared_error",  # Koristi MSE za merenje kvaliteta podele
)

# %%
dtr.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)

# %%
# Metrike
dtr_r2 = r2_score(y_test, y_pred_dtr)
dtr_mae = mean_absolute_error(y_test, y_pred_dtr)
dtr_mse = mean_squared_error(y_test, y_pred_dtr)
dtr_rmse = np.sqrt(dtr_mse)

print(f"Dubina stabla: {dtr.get_depth()}")
print(f"Broj listova: {dtr.get_n_leaves()}")
print(f"R^2 Score: {dtr_r2:.4f}")
print(f"MAE: {dtr_mae:.4f}")
print(f"MSE: {dtr_mse:.4f}")
print(f"RMSE: {dtr_rmse:.4f}")

# %%
y_pred_train = dtr.predict(X_train)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_dtr)

print(f"Train R^2: {train_r2:.4f}")
print(f"Test R^2: {test_r2:.4f}")
print(f"Razlika: {train_r2 - test_r2:.4f}")

# %% [markdown]
# - Zaključci na našem običnom DTR:
#     - R^2 score nam je ~0.997 što znači da nam je model imao odlične rezultate
#     - A MAE, MSE i RMSE su poprilično male, što bi značilo da nam je model dao odlične rezultate
#     - Takodje smo, kako bi proverili da nam model nije slučajno overfit-ovao, uporedili smo R^2 za train i R^2 za test, i možemo zaključiti pošto razlika nije veća od 0.1, da nam je model zapravo dobar

# %% [markdown]
# - Sada ćemo probati da uradimo poredjenje rezultata našeg modela pomoću GridSearchCV
# - Koristićemo param_grid, kako bi to učinili

# %% [markdown]
# - GridSearchCV parametri:
#     - estimator: Model koji želimo da optimizujemo
#     - param_grid: Rečnik parametara za testiranje
#     - cv: Broj fold-ova za cross-validation (5 znači da se podaci dele na 5 delova)
#     - scoring: Metrika za evaluaciju ('r2', 'neg_mean_squared_error', itd.)
#     - n_jobs: Broj CPU jezgara (-1 znači koristi sve dostupne)
#     - verbose: Nivo detaljnosti ispisa (0=tiho, 1=osnovno, 2=detaljno)

# %%
param_grid = {
    "max_depth": [3, 5, 7, 10, 15, 20, None],  # Različite dubine stabla
    "min_samples_split": [2, 5, 10, 20],  # Minimalan broj uzoraka za podelu
    "min_samples_leaf": [1, 2, 4, 8],  # Minimalan broj uzoraka u listu
    "criterion": ["squared_error", "absolute_error"],  # MSE vs MAE
}

# %%
grid_search_dt = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="r2",  # Optimizujemo R^2 score
    n_jobs=-1,  # Koristi sve CPU jezgre
    verbose=1,  # Prikazuj napredak
)

# %%
grid_search_dt.fit(X_train, y_train)

# %%
print(f"Najbolji parametri:")
for param, value in grid_search_dt.best_params_.items():
    print(f"{param}: {value}")

# %%
# Najbolji model
best_dt = grid_search_dt.best_estimator_
y_pred_dt_best = best_dt.predict(X_test)

# Metrike
dt_best_r2 = r2_score(y_test, y_pred_dt_best)
dt_best_mae = mean_absolute_error(y_test, y_pred_dt_best)
dt_best_mse = mean_squared_error(y_test, y_pred_dt_best)
dt_best_rmse = np.sqrt(dt_best_mse)

print(f"Najbolji CV R^2 score: {grid_search_dt.best_score_:.4f}")
print(f"Dubina stabla: {best_dt.get_depth()}")
print(f"Broj listova: {best_dt.get_n_leaves()}")
print(f"Test set performanse:")
print(f"R^2 Score: {dt_best_r2:.4f}")
print(f"MAE: {dt_best_mae:.4f}")
print(f"MSE: {dt_best_mse:.4f}")
print(f"RMSE: {dt_best_rmse:.4f}")

# %% [markdown]
# - Zaključci:
#     - Vidimo da je R^2 score, približan onom od malopre, ali je ipak bolji
#     - Takodje su i MAE, MSE i RMSE manji nego u prethodnom modelu
#     - S' tim što nam je ovde dubina ograničena na 15, dok je u prethodnom bila None, tj. neograničena
#     - U principu najbolji model je ovaj koji je Cross Validation pronašao iz našeg grid-a

# %%
# Vizualizacija važnosti osobina (feature importance)
feature_importance = best_dt.feature_importances_
feature_names = features.columns

# %%
# Sortiranje po važnosti
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(
    range(len(feature_importance)),
    feature_importance[indices],
    color="skyblue",
    edgecolor="black",
)
plt.xticks(
    range(len(feature_importance)),
    [feature_names[i] for i in indices],
    rotation=45,
    ha="right",
)
plt.xlabel("Osobine (Features)", fontsize=12)
plt.ylabel("Važnost", fontsize=12)
plt.title("Važnost osobina u DTR modelu", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %%
print("Rangirane osobine po važnosti:")
for i, idx in enumerate(indices, 1):
    print(f"{i}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

# %% [markdown]
# ## Neural network
# - U našem Neural Network-u imamo sledeće stvari:
#     - Ulazni sloj -> prima podatke
#     - Skriveni sloj -> procesira podatke
#     - Izlazni sloj -> izbacuje rezultate predikcije
# - Dok je Backpropagation algoritam koji izgleda ovako:
#     - Prolaz unapred -> Podaci prolaze kroz NN i računaju se predikcije
#     - Računajanje loss funckije -> Računa se greška izmedju predikcije i stvarne vrednosti
#     - Prolaz unazad -> Gradijent loss se vraća unazad kroz NN
#     - Ažuriranje težina -> Ažuriraju se težine, kako bi smanjili grešku
# - Parametri:
#     - **hidden_layer_sizes**: Arhitektura skrivenih slojeva, npr. (100, 50) = 2 sloja sa 100 i 50 neurona
#     - **activation**: Aktivaciona funkcija ('relu', 'tanh', 'logistic')
#     - **solver**: Algoritam za optimizaciju ('adam', 'sgd', 'lbfgs')
#     - **alpha**: L2 regularizacioni parametar (sprečava overfitting)
#     - **learning_rate_init**: Početna stopa učenja
#     - **max_iter**: Maksimalan broj iteracija (epoha)
#     - **early_stopping**: Zaustavlja treniranje ako se validaciona greška ne poboljšava

# %%
mlp_simple = MLPRegressor(
    hidden_layer_sizes=(20,),  # Jedan skriveni sloj sa 20 neurona
    activation="relu",  # ReLU aktivaciona funkcija: f(x) = max(0, x)
    solver="adam",  # Adam optimizer (Adaptive Moment Estimation)
    alpha=0.0001,  # L2 regularizacioni parametar (default)
    learning_rate_init=0.001,  # Početna stopa učenja
    max_iter=1000,  # Maksimalan broj epoha
    random_state=42,  # Za reproduktivnost
    verbose=False,  # Ne prikazuj napredak
)

# %%
mlp_simple.fit(X_train, y_train)
y_pred_mlp_simple = mlp_simple.predict(X_test)

# %%
# Metrike
mlp_simple_r2 = r2_score(y_test, y_pred_mlp_simple)
mlp_simple_mae = mean_absolute_error(y_test, y_pred_mlp_simple)
mlp_simple_mse = mean_squared_error(y_test, y_pred_mlp_simple)
mlp_simple_rmse = np.sqrt(mlp_simple_mse)

print(f"Arhitektura: {mlp_simple.hidden_layer_sizes}")
print(f"Broj iteracija: {mlp_simple.n_iter_}")
print(f"Loss na kraju treniranja: {mlp_simple.loss_:.6f}")
print(f"R^2 Score: {mlp_simple_r2:.4f}")
print(f"MAE: {mlp_simple_mae:.4f}")
print(f"MSE: {mlp_simple_mse:.4f}")
print(f"RMSE: {mlp_simple_rmse:.4f}")

# %% [markdown]
# - Sada ćemo kao i u DTR-u malopre, da iskoristimo GridCV, kako bi našli najbolji Neural Network

# %%
param_grid_mlp = {
    "hidden_layer_sizes": [
        (50,),  # 1 sloj, 50 neurona
        (100,),  # 1 sloj, 100 neurona
        (50, 50),  # 2 sloja, svaki sa 50 neurona
        (100, 50),  # 2 sloja, 100 i 50 neurona
        (100, 50, 25),  # 3 sloja, 100, 50 i 25 neurona
    ],
    "activation": ["relu", "tanh"],  # ReLU vs tanh aktivacija
    "solver": ["adam"],  # Adam optimizer
    "alpha": [0.0001, 0.001, 0.01],  # Različite vrednosti regularizacije
    "learning_rate_init": [0.001, 0.01],  # Različite stope učenja
}

# %%
grid_search_mlp = GridSearchCV(
    estimator=MLPRegressor(
        max_iter=1000,
        random_state=42,
        early_stopping=True,  # Zaustavlja treniranje ako nema poboljšanja
        validation_fraction=0.1,  # 10% podataka za validaciju
        n_iter_no_change=20,  # Zaustavlja ako nema poboljšanja za 20 iteracija
        verbose=False,
    ),
    param_grid=param_grid_mlp,
    cv=3,  # 3-fold CV (manje fold-ova jer je treniranje sporije)
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

# %%
grid_search_mlp.fit(X_train, y_train)

# %%
best_mlp = grid_search_mlp.best_estimator_
y_pred_mlp_best = best_mlp.predict(X_test)

# Metrike
mlp_best_r2 = r2_score(y_test, y_pred_mlp_best)
mlp_best_mae = mean_absolute_error(y_test, y_pred_mlp_best)
mlp_best_mse = mean_squared_error(y_test, y_pred_mlp_best)
mlp_best_rmse = np.sqrt(mlp_best_mse)

print(f"Best CV R^2 score: {grid_search_mlp.best_score_:.4f}")
print(f"Broj iteracija: {best_mlp.n_iter_}")
print(f"Loss na kraju treniranja: {best_mlp.loss_:.6f}")
print(f"Test set performanse:")
print(f"R^2 Score: {mlp_best_r2:.4f}")
print(f"MAE: {mlp_best_mae:.4f}")
print(f"MSE: {mlp_best_mse:.4f}")
print(f"RMSE: {mlp_best_rmse:.4f}")

# %% [markdown]
# - Zaključci:
#     - R^2 score nam je u oba slučaja visok, tj. u optimalnom slučaju je 0.999, što znači da objašnjava 99.9% varijacije u našem dataset-u
#     - MAE, MSE i RMSE su ekstremno mali što je jako dobar znak
#     - U principu Neural Network odlično radi u našem slučaju
#     - Takodje u odnosu na naš običan NN, imamo i manji broj iteracija u najboljem modelu

# %%
# Prikaz loss curve (kako se greška smanjuje tokom treniranja)
plt.figure(figsize=(12, 5))

# Loss curve za najbolji MLP model
plt.subplot(1, 2, 1)
plt.plot(best_mlp.loss_curve_, linewidth=2)
plt.xlabel("Iteracija (Epoha)", fontsize=12)
plt.ylabel("Loss (greška)", fontsize=12)
plt.title("Loss tokom treniranja (najbolji MLP model)", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)

# Validation curve (ako je early_stopping=True)
if hasattr(best_mlp, "validation_scores_"):
    plt.subplot(1, 2, 2)
    plt.plot(best_mlp.validation_scores_, linewidth=2, color="green")
    plt.xlabel("Iteracija (Epoha)", fontsize=12)
    plt.ylabel("Validation Score", fontsize=12)
    plt.title("Validation Score tokom treniranja", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Poredjenje DTR i NN-a

# %%
results = {
    "Model": [
        "Decision Tree (simple)",
        "Decision Tree (optimized)",
        "Neural Network (20 neurons)",
        "Neural Network (optimized)",
    ],
    "R²": [dtr_r2, dt_best_r2, mlp_simple_r2, mlp_best_r2],
    "MAE": [dtr_mae, dt_best_mae, mlp_simple_mae, mlp_best_mae],
    "MSE": [dtr_mse, dt_best_mse, mlp_simple_mse, mlp_best_mse],
    "RMSE": [dtr_rmse, dt_best_rmse, mlp_simple_rmse, mlp_best_rmse],
}

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("R²", ascending=False).reset_index(drop=True)

# %%
results_df

# %% [markdown]
# - Zaključci i poredjenja modela:
#     - Najbolji model kod nas je naš Neural Network, koji je optimizovan pomoću GridCV-a, u principu tu su izabrani najbolji parametri i sam rezultat je najbolji, tj. R^2 score je najveći i taj model objašnjava 99.99% naše varijanse
#     - Varijansa je merilo koliko se naši podaci medjusobno razlikuju
#     - Npr. R^2 = 1 - (neobjašnjen varijansa / ukupna varijansa)
#     - Takodje su u našem optimizovanom Neural Network-u, naše MAE, MSE i RMSE konkretno male, tako da znamo da model nije pravio greške
#     - Zašto su neki modeli bolji od drugih?
#         - Optimizacija hiperparametara:
#             - Modeli koji su optimizovani (Decision Tree i Neural Network) postižu znatno bolje rezultate od svojih jednostavnih verzija, jer su njihovi parametri prilagođeni strukturi podataka.
#         - Sposobnost modela da uoči nelinearne odnose:
#             - Neural Network modeli su u stanju da modeluju kompleksne i nelinearne zavisnosti u podacima, što je posebno važno za AirQuality dataset, gde odnosi između ulaznih promenljivih nisu striktno linearni.
#         - Kontrola overfitting-a:
#             - Optimizovani modeli bolje balansiraju izmedju prilagodjavanja trening podacima i generalizacije na nove podatke. Dok naši jednostavni modeli mogu ili previše pojednostaviti problem ili se previše prilagoditi podacima
#         - Struktura modela:
#             - Za Decision Tree modeli su intuitivni, ali imaju ograničenja u generalizaciji, dok Neural Network, uz ispravna podešavanja može postići znatno veću preciznost
#     - Na kraju, na osnovu prethodnih zadataka, može se zaključiti da kombinacija složenijeg modela i dobrih hiperparametara i dobra postavka slojeva kod Neural Network-a, mogu značajno poboljšati rezultat našeg modela.

# %% [markdown]
# # Vremenske serije (Faza 7)
# - U ovom odelju ćemo se baviti vremenskim serijama
# - U slučaju mog dataset-a, vremenska serija će biti promena temeprature kroz vreme
# - Koristićemo kolonu DateTime, koja je timestamp i koristićemo Standardizovanu kolonu T, koja predstavlja temperaturu

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

# %%
df["OriginalT"] = temperature_original

# %%
timeseries = df.sort_values("DateTime").set_index("DateTime")["OriginalT"]

# %%
timeseries_daily = timeseries.resample("D").mean()

# %%
timeseries_hourly = timeseries.resample("h").mean().ffill()

# %%
timeseries_daily.head(10)

# %%
timeseries_hourly.head(10)

# %%
plt.figure(figsize=(14, 6))
plt.plot(timeseries_daily, linewidth=2, color="purple")
plt.title("Vremenska serija temperature (T) kroz vreme", fontsize=16, fontweight="bold")
plt.xlabel("Datum i vreme", fontsize=12)
plt.ylabel("Standardizovana Temperatura (T)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# Hajde da izaberemo odgovarajući skup, npr. ceo dataset, pošto je on poredjan po danu pa po satu

# %%
decomposition = seasonal_decompose(
    timeseries_daily.values, model="additive", period=24, extrapolate_trend="freq"
)
decomposition.plot()
plt.suptitle("Dekompozicija vremenske serije", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# Zaključci:
# - Naša temperatura osciluje izmedju ~0 stepeni i ~42 stepena
# - Ima mnogo šuma
# - Vidi se i promena leto - zima
# - I oscijalcije dan - noć takodje
# - Trend:
#     - Ova kriva raste i pada
#     - Kreće se nešto niže u prvom delu, pa raste, pa ponovo pada
#     - Pretpostavka je da je krenula od zime, pa do leta i opet prešla na zimu i proleće u nekom trenutku
#     - Zimi su vrednosti najniže
# - Seasonal:
#     - Pokazuje snažan ponavljajući pattern za svaki od 12 sezona
#     - Ovo nam ukazuje na potrebu za nekim kompleksnijim modelom
# - Residual:
#     - Nasumične oscilacije oko nule
#     - Nema jasan pattern
#     - Ovo su nepredvidivi faktora: kiša, vetar, oluje, oblaci,...
#     - Mogu takodje biti i greške merenja, anomalije i nešto slično

# %% [markdown]
# ## SARIMAX model

# %% [markdown]
# Hajde sad da proverimo stacionarnost pomoću ADF testa:
# - H0: Serija je nestacionarna (ima koren jedinice, prisutan je trend).
# - Ha: Serija je stacionarna.
# - Pravilo odlučivanja:
#     - Ako je P-vrednost ≤ 0.05 (standardni nivo značajnosti), odbacujemo H0​ i zaključujemo da je serija stacionarna.
#     - Ako je P-vrednost > 0.05, ne odbacujemo H0​, serija je nestacionarna.


# %%
def check_stacionarity(timeseries, name):
    print(f"ADF Test za: {name}")
    dftest = adfuller(timeseries.dropna(), autolag="AIC")
    p_value = dftest[1]

    print(f"P-vrednost: {p_value:.4f}")
    if p_value <= 0.05:
        print("=> Zaključak: Serija je STACIONARNA. Nema potrebe za diferenciranjem.")
    else:
        print("=> Zaključak: Serija je NESTACIONARNA. Potrebno je diferenciranje.")


# %%
check_stacionarity(timeseries_daily, "Originalna serija")

# %%
timeseries_daily_diff = timeseries_daily.diff().diff(24).dropna()

# %% [markdown]
# Hajde sada da odredimo p, q, P i Q za ARIMA/SARIMA modele:
# Ovaj korak se sprovodi vizuelnom analizom funkcija korelacije:
# - ACF (Autokorelaciona funkcija): Merenje linearne korelacije serije sa njenim lagovima (prošlim vrednostima).
#     - ACF pomaže u određivanju redova q (MA) i Q (SMA).
#     - Ako ACF "seče" (pada na nulu) posle određenog laga, to je indikacija reda q ili Q.
# - PACF (Parcijalna autokorelaciona funkcija): Merna korelacija serije sa njenim lagovima, eliminšući uticaj međukoraka.
#     - PACF pomaže u određivanju redova p (AR) i P (SAR).
#     - Ako PACF "seče" posle određenog laga, to je indikacija reda p ili P.
# - Ali kako je ovo previše komplikovano za analizu, moraćemo da to uradimo automatski pomoću auto_arima funkcije

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# ACF
plot_acf(
    timeseries_daily_diff.dropna(),
    lags=40,
    ax=axes[0],
    title="Autokorelaciona Funkcija (ACF)",
)
axes[0].set_xlabel("Lag")
# PACF
plot_pacf(
    timeseries_daily_diff.dropna(),
    lags=40,
    ax=axes[1],
    title="Parcijalna Autokorelaciona Funkcija (PACF)",
)
axes[1].set_xlabel("Lag")

plt.suptitle("Identifikacija P i Q za ARIMA/SARIMA", y=1.05)
plt.show()

# %%
train_size = int(len(timeseries_daily) * 0.8)
train_daily = timeseries_daily[:train_size]
test_daily = timeseries_daily[train_size:]

# %% [markdown]
# Sada formiramo naš SARIMAX model:
# - ARIMA (p, d, q)
# - SARIMA (p, d, q) x (P, D, Q, m)
# - SARIMAX (p, d, q) x (P, D, Q, m) + Feature-i (train_daily)

# %%
auto_model = auto_arima(
    train_daily,
    seasonal=True,
    m=7,  # sezonalni period
    max_p=3,
    max_q=3,
    max_P=1,
    max_Q=1,
    max_d=1,
    max_D=1,
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
)

# %%
print(auto_model.summary())
print(f"\nBest model: ARIMA{auto_model.order} x SARIMA{auto_model.seasonal_order}[24]")

# %%
predictions_daily = auto_model.predict(n_periods=len(test_daily))

# %%
plt.figure(figsize=(18, 7))

plt.plot(
    train_daily.index,
    train_daily.values,
    label="Train set",
    linewidth=2,
    color="#06A77D",
    alpha=0.7,
)

plt.plot(
    test_daily.index,
    test_daily.values,
    label="Test set (stvarne)",
    linewidth=2.5,
    color="#2E86AB",
    alpha=0.8,
)

plt.plot(
    test_daily.index,
    predictions_daily,
    label="Predikcije",
    linewidth=2.5,
    color="#F18F01",
    linestyle="--",
    alpha=0.8,
)

plt.axvline(x=test_daily.index[0], color="red", linestyle=":", linewidth=2, alpha=0.7)

plt.title("SARIMAX - Dnevni proseci", fontsize=16, fontweight="bold")
plt.xlabel("Datum", fontsize=13, fontweight="bold")
plt.ylabel("Temperatura (°C)", fontsize=13, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
mae = mean_absolute_error(test_daily, predictions_daily)
rmse = root_mean_squared_error(test_daily, predictions_daily)

print(f"\nPerformanse:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# %% [markdown]
# Zaključci:
# - Vidimo da su rezultati za SARIMAX model katastrofalni
# - Greške su previsoke, a sama predikcija ni blizu tačnog
# - Takodje, to bi trebalo da je tako zato što imamo baš mali broj podataka za treniranje našeg modela
# - Ali kada probamo npr. sa časovnim nizom, dobijam ogromnu potrošnju memorije, ali i to je kada se proba opet, netačno
# - Tako da SARIMAX model nije dobar u ovom slučaju
# - Na osnovu nekog istraživanja na WEB-u, zaključak je da bi Prophet trebalo da uradio posao, ali ga neću implementirati ovde, jer nisam siguran da je potrebno

# %% [markdown]
# ## LSTM - Long Short-Term Memory
# - Tip rekurentne neuronske mreže (RNN)
# - Sposobna da nauči dugotrajne zavisnosti u vremenskim serijama
# - LSTM bi trebalo da može bolje da hvata nelinearne obrasce
# - Bolje skalira sa velikim dataset-om
# - Ovde ćemo koristiti timeseries po satu
# - LSTM lako barata sa velikim datasetom

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(42)
np.random.seed(42)

# %%
timeseries_lstm = timeseries_hourly.copy()

# %%
print(f"Ukupno podataka: {len(timeseries_lstm)} sati")
print(f"Period: {timeseries_lstm.index[0]} do {timeseries_lstm.index[-1]}")
print(f"Min temperatura: {timeseries_lstm.min():.2f}°C")
print(f"Max temperatura: {timeseries_lstm.max():.2f}°C")

# %%
data = timeseries_lstm.values.reshape(-1, 1)

# %%
print(f"Shape podataka: {data.shape}")

# %% [markdown]
# Moramo da uradimo skaliranje, jer Neural Network radi sa vrednostima od 0 do 1

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# %%
def create_sequences(data, seq_length, forecast_horizon=1):
    """
    data: Normalizovani podaci
    seq_length: Dužina ulazne sekvence (npr. 168 sati = 7 dana)
    forecast_horizon: Koliko koraka unapred predviđamo (default 1)

    X: Input sekvence shape (samples, seq_length, 1)
    y: Target vrednosti shape (samples, forecast_horizon)
    """
    X, y = [], []

    for i in range(len(data) - seq_length - forecast_horizon + 1):
        # Input: prethodnih seq_length vrednosti
        X.append(data[i : i + seq_length])

        # Output: sledećih forecast_horizon vrednosti
        if forecast_horizon == 1:
            y.append(data[i + seq_length])
        else:
            y.append(data[i + seq_length : i + seq_length + forecast_horizon])

    return np.array(X), np.array(y)


# %%
SEQ_LENGTH = 24 * 7  # 7 dana = 168 sati
FORECAST_HORIZON = 1  # Predviđamo 1 sat unapred

print(f"Dužina sekvence: {SEQ_LENGTH} sati ({SEQ_LENGTH/24:.0f} dana)")
print(f"Predvidjamo: {FORECAST_HORIZON} sat unapred")

X, y = create_sequences(scaled_data, SEQ_LENGTH, FORECAST_HORIZON)

print(f"X shape: {X.shape}")  # (samples, 168, 1)
print(f"y shape: {y.shape}")  # (samples, 1)
print(f"Ukupno sekvenci: {len(X)}")

# %%
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size : train_size + val_size]
y_val = y[train_size : train_size + val_size]

X_test = X[train_size + val_size :]
y_test = y[train_size + val_size :]

print("Dataset split:")
print(f"Train: {len(X_train)} sekvenci ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val: {len(X_val)} sekvenci ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test)} sekvenci ({len(X_test)/len(X)*100:.1f}%)")

# Proveri datume
train_end_idx = train_size + SEQ_LENGTH
val_end_idx = train_size + val_size + SEQ_LENGTH
test_end_idx = len(timeseries_lstm)

print(
    f"Trening period: {timeseries_lstm.index[SEQ_LENGTH]} do {timeseries_lstm.index[train_end_idx-1]}"
)
print(
    f"Val period: {timeseries_lstm.index[train_end_idx]} do {timeseries_lstm.index[val_end_idx-1]}"
)
print(
    f"Test period: {timeseries_lstm.index[val_end_idx]} do {timeseries_lstm.index[-1]}"
)


# %%
def build_lstm_model(seq_length, n_features=1):
    model = keras.Sequential(
        [
            keras.Input(shape=(seq_length, n_features)),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(FORECAST_HORIZON),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",  # Mean Squared Error
        metrics=["mae"],  # Mean Absolute Error
    )

    return model


# %%
model = build_lstm_model(SEQ_LENGTH)

# %%
early_stop = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

# %%
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1
)

callbacks = [early_stop, reduce_lr]

# %%
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

# %%
y_pred_scaled = model.predict(X_test, verbose=0)

# Vraćamo skalirane vrednosti nazad u originalni opseg
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# %%
mse_lstm = mean_squared_error(y_test_actual, y_pred)
mae_lstm = mean_absolute_error(y_test_actual, y_pred)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_test_actual, y_pred)

print(f"LSTM Model Performanse na test timeseries-u, tj. setu:")
print(f"R^2 Score: {r2_lstm:.4f}")
print(f"MAE: {mae_lstm:.4f}")
print(f"MSE: {mse_lstm:.4f}")
print(f"RMSE: {rmse_lstm:.4f}")

# %% [markdown]
# Zaključak:
# - Naš NN objašnjava 91.9% varijanse
# - Svakako model nije savršen
# - Greške su mu poprilično visoke, tako da znamo da nije najbolji

# %%
test_start_idx = train_size + val_size + SEQ_LENGTH
test_dates = timeseries_lstm.index[test_start_idx : test_start_idx + len(y_test)]

# %%
plt.figure(figsize=(18, 7))

# Train
train_dates = timeseries_lstm.index[SEQ_LENGTH:train_end_idx]
plt.plot(
    train_dates,
    timeseries_lstm.values[SEQ_LENGTH:train_end_idx],
    label="Train set",
    linewidth=1.5,
    color="#06A77D",
    alpha=0.5,
)

# Validation
val_dates = timeseries_lstm.index[train_end_idx:val_end_idx]
plt.plot(
    val_dates,
    timeseries_lstm.values[train_end_idx:val_end_idx],
    label="Validation set",
    linewidth=1.5,
    color="#F18F01",
    alpha=0.5,
)

# Test - stvarne vrednosti
plt.plot(
    test_dates,
    y_test_actual,
    label="Test set (stvarne vrednosti)",
    linewidth=2.5,
    color="#2E86AB",
    alpha=0.8,
)

# Test - predikcije
plt.plot(
    test_dates,
    y_pred,
    label="LSTM predikcije",
    linewidth=2.5,
    color="#A23B72",
    linestyle="--",
    alpha=0.8,
)

# Vertikalne linije
plt.axvline(
    x=train_dates[-1],
    color="red",
    linestyle=":",
    linewidth=2,
    alpha=0.6,
    label="Train/Val split",
)
plt.axvline(
    x=val_dates[-1],
    color="orange",
    linestyle=":",
    linewidth=2,
    alpha=0.6,
    label="Val/Test split",
)

# Stilizovanje
plt.title(
    "LSTM Model - Kompletna vremenska serija i predikcije",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Datum i vreme", fontsize=13, fontweight="bold")
plt.ylabel("Temperatura (°C)", fontsize=13, fontweight="bold")
plt.legend(fontsize=10, loc="best", ncol=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# Zaključak:
# - Sa grafika vidimo da LSTM približno prati test seriju
# - Ali opet iz metrika vidimo da model nije najbolji
# - Svakako je bolji od SARIMAX modela, koji nije mogao da razume ove podatke
# - Ovo je NN sa 5 layera, jednim ulaznim, jednim izlaznim i 3 skrivena sloja
# - Od 3 skrivena sloja, najbitniji je LSTM sloj, on je jezgro našeg NN-a
# - LSTM ima 64 "memorijska kanala" u našem slučaju, može imati i više ali je rizik da se desi overfitting
# - Input layer, nam govori samo koliko koraka gledamo u nazad i koliki je broj varijabli po koraku
# - Dropout nam pomaže da nam se model ne uhvati za šum u podacima, i takodje nasumično gasi, u ovom slučaju 20% neurona, kako bi sprečio ovo prvo
# - Koristimo RELU, kao aktivacionu funkciju
# - Za output sloj, od FORECAST_HORIZON-a nam zavisi koliko dana hoćemo da predvidimo
# - Nema aktivacija u tom sloju

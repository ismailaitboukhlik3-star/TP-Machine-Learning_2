
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler
)

df = pd.read_csv("transactions_TP2.csv")

print(df.head())

print("Dimensions du dataset :", df.shape)

print(df.dtypes)


nb_manquantes = df.isnull().sum()


pourcentage = (df.isnull().mean() * 100)


resultat = pd.DataFrame({
    "Nombre_manquantes": nb_manquantes,
    "Pourcentage (%)": pourcentage
})


resultat = resultat.sort_values(
    by="Pourcentage (%)",
    ascending=False
)


print(resultat.round(2))




nb_doublons = df.duplicated(subset=["id_operation"]).sum()

print("Nombre de doublons :", nb_doublons)


df = df.drop_duplicates(
    subset=["id_operation"],
    keep="first"
)


print("Nombre de lignes restantes :", len(df))








montants = df["montant_dh"].dropna()


plt.figure(figsize=(8, 5))
plt.hist(montants, bins=30, edgecolor="black")
plt.title("Distribution des montants des transactions")
plt.xlabel("Montant (DH)")
plt.ylabel("Fréquence")
plt.grid(axis="y", alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot(montants)
plt.title("Boxplot des montants des transactions")
plt.ylabel("Montant (DH)")
plt.show()


print("Les 5 montants les plus élevés :")
print(montants.nlargest(5))


effectifs = df["fraude"].value_counts().sort_index()

print("Nombre de transactions par classe :")
print(effectifs)


proportions = df["fraude"].value_counts(
    normalize=True
).sort_index() * 100

print("\nProportions en pourcentage :")
print(proportions)


plt.figure(figsize=(7, 5))

plt.bar(
    ["Normale (0)", "Fraude (1)"],
    effectifs.values,
    edgecolor="black"
)

plt.title("Répartition des transactions")
plt.xlabel("Classes")
plt.ylabel("Nombre de transactions")

plt.show()


colonnes_a_exclure = [
    "id_operation",
    "id_client",
    "resultat_enquete"
]

print(df[colonnes_a_exclure].head())


y = df["fraude"]


X = df.drop(
    columns=[
        "id_operation",
        "id_client",
        "resultat_enquete",
        "fraude"
    ]
)


print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

print("\nVariables de X :")
print(X.columns.tolist())


X_ref, X_ver, y_ref, y_ver = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Groupe de référence :", X_ref.shape)
print("Groupe de vérification :", X_ver.shape)


print("\nProportion dans le groupe de référence :")
print(y_ref.value_counts(normalize=True) * 100)

print("\nProportion dans le groupe de vérification :")
print(y_ver.value_counts(normalize=True) * 100)



variables_numeriques = X.select_dtypes(
    include=["number"]
).columns.tolist()

variables_categorielles = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("Variables numériques :")
print(variables_numeriques)

print("\nVariables catégorielles :")
print(variables_categorielles)


pipeline_num = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


X_ref_num = pipeline_num.fit_transform(
    X_ref[variables_numeriques]
)

X_ver_num = pipeline_num.transform(
    X_ver[variables_numeriques]
)


print("Groupe de référence :", X_ref_num.shape)
print("Groupe de vérification :", X_ver_num.shape)


print("\nMédianes utilisées :")
print(pipeline_num.named_steps["imputer"].statistics_)


print("\nÉchelles utilisées :")
print(pipeline_num.named_steps["scaler"].scale_)

SimpleImputer(strategy="most_frequent")

pipeline_cat = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


X_ref_cat = pipeline_cat.fit_transform(
    X_ref[variables_categorielles]
)


X_ver_cat = pipeline_cat.transform(
    X_ver[variables_categorielles]
)


print("Groupe de référence :", X_ref_cat.shape)
print("Groupe de vérification :", X_ver_cat.shape)

print("\nNouvelles variables :")
print(pipeline_cat.named_steps["encoder"].get_feature_names_out())


preprocesseur = ColumnTransformer(
    transformers=[
        ("num", pipeline_num, variables_numeriques),
        ("cat", pipeline_cat, variables_categorielles)
    ]
)


X_ref_transforme = preprocesseur.fit_transform(X_ref)


X_ver_transforme = preprocesseur.transform(X_ver)


print("Groupe de référence :",
      X_ref_transforme.shape)

print("Groupe de vérification :",
X_ver_transforme.shape)


pipeline_standard = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


pipeline_robust = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])


X_std = pipeline_standard.fit_transform(
    X_ref[variables_numeriques]
)


Q1 = X_ref["montant_dh"].quantile(0.25)


Q3 = X_ref["montant_dh"].quantile(0.75)


IQR = Q3 - Q1


print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)



montants = X_ref["montant_dh"].dropna()


Q1 = montants.quantile(0.25)
Q3 = montants.quantile(0.75)


IQR = Q3 - Q1

limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR


outliers = montants[
    (montants < limite_inf) |
    (montants > limite_sup)
]


print("Limite inférieure :", limite_inf)
print("Limite supérieure :", limite_sup)

print("Nombre de valeurs extrêmes :", len(outliers))

print("\nQuelques valeurs extrêmes :")
print(outliers.sort_values(ascending=False).head(10))




montants = X_ref["montant_dh"].dropna()


Q1 = montants.quantile(0.25)
Q3 = montants.quantile(0.75)


IQR = Q3 - Q1


limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR


montants_originaux = montants.copy()


montants_limites = montants.clip(
    lower=limite_inf,
    upper=limite_sup
)


plt.figure(figsize=(10, 5))

plt.boxplot([
    montants_originaux,
    montants_limites
])

plt.xticks(
    [1, 2],
    ["Montants originaux", "Montants limités"]
)

plt.title("Comparaison des valeurs extrêmes")
plt.ylabel("Montant (DH)")
plt.grid(axis="y", alpha=0.3)

plt.show()


print("Maximum original :", montants_originaux.max())
print("Maximum après Capping :", montants_limites.max())

print("Nombre de valeurs limitées :",
      (montants_originaux > limite_sup).sum())




scalers = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}

resultats = {}


for nom, scaler in scalers.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    ref = pipeline.fit_transform(
        X_ref[["montant_dh"]]
    )

   
    ver = pipeline.transform(
        X_ver[["montant_dh"]]
    )

   
    resultats[nom] = {
        "reference": ref,
        "verification": ver
    }

    print("\nMéthode :", nom)
    print("Référence :", ref.shape)
    print("Vérification :", ver.shape)

montants = X_ref["montant_dh"].dropna()


Q1 = montants.quantile(0.25)
Q2 = montants.quantile(0.50)
Q3 = montants.quantile(0.75)


m1 = montants.loc[(montants - Q1).abs().idxmin()]
m2 = montants.loc[(montants - Q2).abs().idxmin()]
m3 = montants.loc[(montants - Q3).abs().idxmin()]


m4 = montants.max()


exemples = pd.DataFrame({
    "montant_dh": [m1, m2, m3, m4]
})

print("Valeurs initiales :")
print(exemples)


methodes = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}


resultats = {}

for nom, scaler in methodes.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    
    pipeline.fit(X_ref[["montant_dh"]])

    
    valeurs = pipeline.transform(exemples)

    resultats[nom] = valeurs.flatten()


tableau = pd.DataFrame(resultats)

tableau.insert(
    0,
    "Valeur initiale (DH)",
    exemples["montant_dh"].values
)

print("\nComparaison des transformations :")
print(tableau.round(4))


methodes = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}


for nom, scaler in methodes.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])


    montants_transformes = pipeline.fit_transform(
        X_ref[["montant_dh"]]
    )

   
    valeurs = montants_transformes.flatten()

   
    plt.figure(figsize=(8, 5))

    plt.hist(valeurs, bins=30, edgecolor="black")

    plt.title("Histogramme avec " + nom)
    plt.xlabel("Montant transformé")
    plt.ylabel("Fréquence")

    plt.grid(axis="y", alpha=0.3)
    plt.show()




pipeline_minmax = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])


pipeline_minmax.fit(X_ref[["montant_dh"]])


nouvelle_transaction = pd.DataFrame({
    "montant_dh": [40000]
})


resultat = pipeline_minmax.transform(
    nouvelle_transaction
)


print("Montant initial :", 40000, "DH")
print("Montant transformé :", resultat[0][0])


effectifs = y_ref.value_counts().sort_index()

print("Nombre de transactions par classe :")
print(effectifs)


rapport = effectifs[0] / effectifs[1]

print("\nRapport entre les classes :")
print(rapport)


proportions = y_ref.value_counts(
    normalize=True
).sort_index() * 100

print("\nProportions :")
print(proportions)




print("Avant le suréchantillonnage :")
print(y_ref.value_counts().sort_index())


ros = RandomOverSampler(random_state=42)


X_ref_over, y_ref_over = ros.fit_resample(
    X_ref,
    y_ref
)


print("\nAprès le suréchantillonnage :")
print(y_ref_over.value_counts().sort_index())


print("\nNombre de lignes avant :", len(y_ref))
print("Nombre de lignes après :", len(y_ref_over))


print("Avant le sous-échantillonnage :")
print(y_ref.value_counts().sort_index())


rus = RandomUnderSampler(random_state=42)


X_ref_under, y_ref_under = rus.fit_resample(
    X_ref,
    y_ref
)


print("\nAprès le sous-échantillonnage :")
print(y_ref_under.value_counts().sort_index())


print("\nNombre de lignes avant :", len(y_ref))
print("Nombre de lignes après :", len(y_ref_under))

supprimees = len(y_ref) - len(y_ref_under)

print("Nombre de lignes supprimées :", supprimees)




A = np.array([100, 10])
B = np.array([200, 20])


lambda_ = 0.4


C = A + lambda_ * (B - A)

print("Transaction A :", A)
print("Transaction B :", B)
print("Transaction synthétique C :", C)

avant = y_ref.value_counts().sort_index()

apres_over = y_ref_over.value_counts().sort_index()


apres_under = y_ref_under.value_counts().sort_index()


comparaison = pd.DataFrame({
    "Avant reequilibrage": avant,
    "Apres OverSampler": apres_over,
    "Apres UnderSampler": apres_under
})

comparaison.index = ["Normale (0)", "Fraude (1)"]


print(comparaison)


print("\nGroupe de verification (non modifie) :")
print(y_ver.value_counts().sort_index())


print("\nTotal avant :", len(y_ref))
print("Total OverSampler :", len(y_ref_over))
print("Total UnderSampler :", len(y_ref_under))
print("Total verification :", len(y_ver))


pipeline_num_final = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])


pipeline_cat_final = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocesseur_final = ColumnTransformer(
    transformers=[
        ("num", pipeline_num_final, variables_numeriques),
        ("cat", pipeline_cat_final, variables_categorielles)
    ]
)


print(preprocesseur_final)




X_ref_final = preprocesseur_final.fit_transform(X_ref)


X_ver_final = preprocesseur_final.transform(X_ver)


print("Groupe de référence :")
print(X_ref_final.shape)

print("\nGroupe de vérification :")
print(X_ver_final.shape)

print("\nValeurs manquantes dans le groupe de référence :")
print(np.isnan(X_ref_final).sum())

print("\nValeurs manquantes dans le groupe de vérification :")
print(np.isnan(X_ver_final).sum())


nouvelle_transaction = X_ref.iloc[[0]].copy()


nouvelle_transaction["canal"] = "canal_inconnu_test"

print("Nouvelle transaction :")
print(nouvelle_transaction[["canal"]])


resultat = preprocesseur_final.transform(
    nouvelle_transaction
)


print("\nTransformation réussie !")
print("Dimensions :", resultat.shape)


if sparse.issparse(resultat):
    resultat = resultat.toarray()

noms = preprocesseur_final.get_feature_names_out()

indices_canal = [
    i for i, nom in enumerate(noms)
    if "canal_" in nom
]

print("\nEncodage de la catégorie inconnue :")

print(pd.DataFrame(
    resultat[:, indices_canal],
    columns=noms[indices_canal]
))
joblib.dump(
    preprocesseur_final,
    "preprocesseur_final.joblib"
)

print("Préprocesseur enregistré avec succès !")


preprocesseur_charge = joblib.load(
    "preprocesseur_final.joblib"
)

print("Préprocesseur rechargé avec succès !")


echantillon = X_ref.iloc[:5]


resultat_original = preprocesseur_final.transform(
    echantillon
)


resultat_charge = preprocesseur_charge.transform(
    echantillon
)


if sparse.issparse(resultat_original):
    resultat_original = resultat_original.toarray()

if sparse.issparse(resultat_charge):
    resultat_charge = resultat_charge.toarray()


print("\nRésultats identiques ?")
print(np.allclose(resultat_original, resultat_charge))

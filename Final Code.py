import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import requests
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, auc,
    confusion_matrix, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

RDLogger.DisableLog('rdApp.*')


url = "https://raw.githubusercontent.com/Money2107/Capstone-Project-5520/main/imatinib_Cleaned_Dataset.csv"
df = pd.read_csv(url, sep=';', quotechar='"', engine='python', on_bad_lines='skip')


df.columns = df.columns.str.strip('"')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip('"')

print(f"Original data shape: {df.shape}")


df_ic50 = df[df['Standard Type'] == 'IC50'].copy()
df_ic50 = df_ic50.dropna(subset=['Smiles', 'Standard Value'])
df_ic50['Standard Value'] = pd.to_numeric(df_ic50['Standard Value'], errors='coerce')
df_ic50 = df_ic50.dropna(subset=['Standard Value'])

print(f"Filtered IC50 data shape: {df_ic50.shape}")



def mol_to_fp(mol):
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    else:
        return None


df_ic50['mol'] = df_ic50['Smiles'].apply(Chem.MolFromSmiles)
df_ic50['fingerprint'] = df_ic50['mol'].apply(mol_to_fp)
df_ic50 = df_ic50[df_ic50['fingerprint'].notnull()].copy()


def fp_to_array(fp):
    arr = np.zeros((2048,), dtype=int)
    if fp is not None:
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return np.nan


df_ic50['fp_array'] = df_ic50['fingerprint'].apply(fp_to_array)



def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta = response.text
        seq = ''.join(fasta.split('\n')[1:]).strip()
        return seq
    else:
        return None


unique_targets = df_ic50['Target ChEMBL ID'].str.strip().unique()
target_sequences = {}

for target_id in unique_targets:
    url = f'https://www.ebi.ac.uk/chembl/api/data/target/{target_id}.json'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            components = data.get('target_components', [])
            if components and 'accession' in components[0]:
                uniprot_id = components[0]['accession']
                seq = fetch_uniprot_sequence(uniprot_id)
                target_sequences[target_id] = seq
            else:
                target_sequences[target_id] = None
        except Exception:
            target_sequences[target_id] = None
    else:
        target_sequences[target_id] = None

df_ic50['Protein Sequence'] = df_ic50['Target ChEMBL ID'].str.strip().map(target_sequences)

print(f"Sequences fetched for {df_ic50['Protein Sequence'].notnull().sum()} out of {len(df_ic50)} entries")


amino_acids = list("ACDEFGHIKLMNPQRSTVWY")


def aa_composition(seq):
    seq = seq.upper()
    length = len(seq)
    comp = [seq.count(aa) / length if length > 0 else 0 for aa in amino_acids]
    return np.array(comp)


df_ic50['protein_feat'] = df_ic50['Protein Sequence'].apply(
    lambda x: aa_composition(x) if isinstance(x, str) else np.nan)


df_clean = df_ic50.dropna(subset=['fp_array', 'protein_feat']).copy()

X_mol = np.stack(df_clean['fp_array'].values)
X_prot = np.stack(df_clean['protein_feat'].values)
X = np.hstack([X_mol, X_prot])


y = (df_clean['Standard Value'].astype(float) < 1000).astype(int)

print(f"Final dataset size: {X.shape[0]} samples, {X.shape[1]} features each")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

sns.set_theme(style="whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#ff4d4d", "#4da6ff", "#33cc33", "#ff9933", "#cc33ff", "#00cccc"
])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y_train, ax=axes[0], palette=["#ff4d4d", "#4da6ff"])
axes[0].set_title("Before SMOTE", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

sns.countplot(x=y_train_res, ax=axes[1], palette=["#33cc33", "#ff9933"])
axes[1].set_title("After SMOTE", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)
y_pred_rf = clf.predict(X_test)
y_proba_rf = clf.predict_proba(X_test)[:, 1]

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf'],
    'class_weight': [None, 'balanced']
}

svm = SVC(probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("Best SVM Parameters:", grid.best_params_)

best_svm = grid.best_estimator_
y_pred_svm = best_svm.predict(X_test)
y_proba_svm = best_svm.predict_proba(X_test)[:, 1]

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_svm))


def plot_confusion_matrix_bright(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral", cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.show()



def plot_roc_curves_bold(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_feature_importance(model, n_features=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]

    plt.figure(figsize=(10, 6))
    plt.bar(range(n_features), importances[indices], color='r', align='center')
    plt.xticks(range(n_features), indices)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.show()


def plot_precision_recall_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, lw=3, label=f"{name} (AP = {avg_precision:.2f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=15, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()



models = {'Random Forest': clf, 'SVM': best_svm}
plot_roc_curves_bold(models, X_test, y_test)


plot_confusion_matrix_bright(y_test, y_pred_rf, "Random Forest Confusion Matrix")
plot_confusion_matrix_bright(y_test, y_pred_svm, "SVM Confusion Matrix")


plot_feature_importance(clf, n_features=20)


plot_precision_recall_curves(models, X_test, y_test)

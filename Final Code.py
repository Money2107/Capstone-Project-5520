#%%
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import requests
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Disable RDKit warnings for clean output
RDLogger.DisableLog('rdApp.*')

#%% Load and clean data
file_path = r"C:\Users\aryan\Downloads\DOWNLOAD-123PGkwNz0F5D2N1bOH-dO2T4_X6xWvMO7n8RNk-wRs_eq_\imatinib(2).csv"

df = pd.read_csv(
    file_path,
    sep=';',
    quotechar='"',
    engine='python',
    on_bad_lines='skip'
)

# Strip quotes from column names
df.columns = df.columns.str.strip('"')

# Strip quotes from string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip('"')

print(f"Original data shape: {df.shape}")
print(df.columns)
print(df.head())

# Check missing values
print("Missing values per column:\n", df.isnull().sum())

# Filter IC50 data, use .copy() to avoid SettingWithCopyWarning
df_ic50 = df[df['Standard Type'] == 'IC50'].copy()

# Drop rows with missing 'Smiles' or 'Standard Value'
df_ic50 = df_ic50.dropna(subset=['Smiles', 'Standard Value'])

# Convert 'Standard Value' to numeric and drop NaNs
df_ic50['Standard Value'] = pd.to_numeric(df_ic50['Standard Value'], errors='coerce')
df_ic50 = df_ic50.dropna(subset=['Standard Value'])

print(f"Filtered IC50 data shape: {df_ic50.shape}")

#%% Generate Morgan fingerprints (2048 bits)
def mol_to_fp(mol):
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    else:
        return None

# Convert SMILES to molecules and fingerprints
df_ic50['mol'] = df_ic50['Smiles'].apply(Chem.MolFromSmiles)
df_ic50['fingerprint'] = df_ic50['mol'].apply(mol_to_fp)

# Remove entries with None fingerprints
df_ic50 = df_ic50[df_ic50['fingerprint'].notnull()].copy()

print(f"After fingerprint filtering: {df_ic50.shape}")

# Convert fingerprints to numpy arrays for ML
def fp_to_array(fp):
    arr = np.zeros((2048,), dtype=int)
    if fp is not None:
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return np.nan

df_ic50['fp_array'] = df_ic50['fingerprint'].apply(fp_to_array)

print(df_ic50[['Smiles', 'fp_array']].head())

#%% Fetch protein sequences from ChEMBL and UniProt
def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta = response.text
        seq = ''.join(fasta.split('\n')[1:]).strip()
        return seq
    else:
        return None

# Get unique target IDs and initialize dictionary
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

# Map sequences to dataframe
df_ic50['Protein Sequence'] = df_ic50['Target ChEMBL ID'].str.strip().map(target_sequences)

print(f"Sequences fetched for {df_ic50['Protein Sequence'].notnull().sum()} out of {len(df_ic50)} entries")

#%% Calculate amino acid composition features
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def aa_composition(seq):
    seq = seq.upper()
    length = len(seq)
    comp = [seq.count(aa) / length if length > 0 else 0 for aa in amino_acids]
    return np.array(comp)

df_ic50['protein_feat'] = df_ic50['Protein Sequence'].apply(lambda x: aa_composition(x) if isinstance(x, str) else np.nan)

# Drop rows with missing fingerprint or protein features
df_clean = df_ic50.dropna(subset=['fp_array', 'protein_feat']).copy()

# Prepare features and labels
X_mol = np.stack(df_clean['fp_array'].values)
X_prot = np.stack(df_clean['protein_feat'].values)
X = np.hstack([X_mol, X_prot])

# Binary label: active if IC50 < 1000 nM, else inactive
y = (df_clean['Standard Value'].astype(float) < 1000).astype(int)

print(f"Final dataset size: {X.shape[0]} samples, {X.shape[1]} features each")

#%% Train and evaluate Random Forest with SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

#%% Train and evaluate SVM with GridSearchCV and SMOTE
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf'],
    'class_weight': [None, 'balanced']
}

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

svm = SVC(probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("Best SVM Parameters:", grid.best_params_)

best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)
y_proba = best_svm.predict_proba(X_test)[:, 1]

print("SVM Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

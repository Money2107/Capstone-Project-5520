from chembl_webresource_client.new_client import new_client
import pandas as pd
import requests

# Step 1: Set ChEMBL ID for Imatinib
imatinib_chembl_id = "CHEMBL941"

# Step 2: Get IC50 bioactivity data for human targets
activity_client = new_client.activity
activities = activity_client.filter(
    molecule_chembl_id=imatinib_chembl_id,
    standard_type="IC50",
    target_organism="Homo sapiens"
)
df = pd.DataFrame(activities)

# Step 3: Get unique ChEMBL target IDs
target_ids = df['target_chembl_id'].dropna().unique()

# Step 4: Retrieve UniProt accessions and FASTA sequences
target_client = new_client.target
fasta_sequences = {}

for target_id in target_ids:
    try:
        target_info = target_client.get(target_id)
        components = target_info.get('target_components', [])
        if components:
            accession = components[0].get('accession')
            if accession:
                fasta_url = f"https://www.uniprot.org/uniprot/{accession}.fasta"
                response = requests.get(fasta_url)
                if response.status_code == 200:
                    fasta_sequences[accession] = response.text
    except Exception as e:
        print(f"Error processing target {target_id}: {e}")

# Step 5: Save FASTA sequences to file
with open("imatinib_human_targets.fasta", "w") as file:
    for sequence in fasta_sequences.values():
        file.write(sequence)

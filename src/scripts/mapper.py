import pickle
import pandas as pd
import re
import json

def normalize_name(name):
    """Lowercase, strip punctuation, spaces for name harmonization."""
    if not isinstance(name, str): return ''
    return re.sub(r'[\W_]+', '', name.lower().strip())

def flatten_synonyms(syns):
    """Synonyms may be list-of-lists or strings separated by |, /, ;."""
    if not syns: return set()
    if isinstance(syns, str):
        # Split by common separators
        return set(s.strip() for s in re.split(r'[|/;,]', syns) if s.strip())
    if isinstance(syns, (list, set)):
        res = set()
        for s in syns:
            res.update(flatten_synonyms(s))
        return res
    return set()

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def force_list(x):
    return x if isinstance(x, list) else [x] if x is not None else []

def build_drug_mappings(drugbank_path, drugs_parquet_path):
    # Load DrugBank list-of-dict (pickled from DrugBankParser)
    drugbank = load_pickle(drugbank_path)
    # Load Open Targets drugs as DataFrame (example parquet file)
    ot_drugs = pd.read_parquet(drugs_parquet_path)
    # Indexes
    drugbank_by_id = {}
    drugbank_by_inchikey = {}
    drugbank_by_cas = {}
    drugbank_by_name = {}
    synonym_to_id = {}

    for entry in drugbank:
        dbid = entry.get('drugbank_id')
        name = entry.get('name')
        inchikey = entry.get('inchikey')
        cas = entry.get('cas_number')
        syns = flatten_synonyms(entry.get('synonyms', []))

        if dbid: drugbank_by_id[dbid] = entry
        if inchikey: drugbank_by_inchikey[inchikey] = dbid
        if cas: drugbank_by_cas[cas] = dbid
        if name:
            nn = normalize_name(name)
            drugbank_by_name[nn] = dbid
            synonym_to_id[nn] = dbid
        for syn in syns:
            ns = normalize_name(syn)
            if ns and ns not in synonym_to_id:
                synonym_to_id[ns] = dbid

    # ChEMBL & Open Targets
    chembl_to_db = {}
    for ix, row in ot_drugs.iterrows():
        chembl = row.get('id') or row.get('chembl_id')
        # Try matching by name/synonym/inchikey
        dbid = None
        row_name = row.get('name', '')
        row_syns = flatten_synonyms(row.get('synonyms', []))
        keys = [normalize_name(row_name)] + [normalize_name(s) for s in row_syns]
        for k in keys:
            dbid = synonym_to_id.get(k) or dbid
        if chembl and dbid:
            chembl_to_db[chembl] = dbid

    # Return all key indices for lookup
    return {
        'by_id': drugbank_by_id,
        'by_inchikey': drugbank_by_inchikey,
        'by_cas': drugbank_by_cas,
        'by_name': drugbank_by_name,
        'synonym_to_id': synonym_to_id,
        'chembl_to_db': chembl_to_db
    }

def build_disease_mappings(mesh_pickle_path, ot_diseases_parquet):
    mesh_data = load_pickle(mesh_pickle_path)
    mesh_desc = mesh_data['descriptors']
    # MeSH: map by UI, by name, by synonyms
    mesh_by_id = {}
    mesh_by_name = {}
    mesh_synonym_to_id = {}
    for d in mesh_desc.values():
        mesh_by_id[d['id']] = d
        nn = normalize_name(d['name'])
        mesh_by_name[nn] = d['id']
        mesh_synonym_to_id[nn] = d['id']
        for syn in flatten_synonyms(d.get('synonyms', [])):
            ns = normalize_name(syn)
            mesh_synonym_to_id[ns] = d['id']

    # Open Targets
    df = pd.read_parquet(ot_diseases_parquet)
    ot_by_efo = {}
    ot_by_name = {}
    ot_mesh_to_efo = {}
    for ix, row in df.iterrows():
        efo_id = row.get('id') or row.get('efo_id')
        name = row.get('name', '')
        syns = flatten_synonyms(row.get('synonyms', []))
        mesh_xrefs = set(x for x in force_list(row.get('dbXRefs', [])) if x.startswith('MESH:'))
        # Name-based
        ot_by_efo[efo_id] = row
        ot_by_name[normalize_name(name)] = efo_id
        for syn in syns:
            ot_by_name[normalize_name(syn)] = efo_id
        # Cross-ref based
        for mx in mesh_xrefs:
            mesh_id = mx.split(':')[1]
            ot_mesh_to_efo[mesh_id] = efo_id

    return {
        'mesh_by_id': mesh_by_id,
        'mesh_by_name': mesh_by_name,
        'mesh_synonym_to_id': mesh_synonym_to_id,
        'ot_by_efo': ot_by_efo,
        'ot_by_name': ot_by_name,
        'mesh_to_efo': ot_mesh_to_efo
    }

def build_target_mappings(target_pickle_path, ot_targets_parquet):
    # Load whatever format you pickled for "target" details (change field names & paths as needed)
    targets_data = load_pickle(target_pickle_path)
    ot_targets = pd.read_parquet(ot_targets_parquet)
    ensembl_to_hgnc = {}
    ensembl_to_uniprot = {}
    symbol_to_id = {}

    for t in targets_data:
        ens = t.get("ensembl_id") or t.get("id")
        hgnc = t.get("hgnc_id") or t.get("symbol")
        uniprot = t.get("uniprot") or t.get("uniprot_id")
        symbol = t.get("name") or t.get("symbol")
        if ens:
            if hgnc: ensembl_to_hgnc[ens] = hgnc
            if uniprot: ensembl_to_uniprot[ens] = uniprot
        if symbol: symbol_to_id[normalize_name(symbol)] = ens

    ot_ensembls = set()
    for ix, row in ot_targets.iterrows():
        ens = row.get("id")
        symbol = row.get("approvedSymbol") or row.get("symbol")
        if ens: ot_ensembls.add(ens)
        if symbol:
            symbol_to_id[normalize_name(symbol)] = ens

    return {
        'ensembl_to_hgnc': ensembl_to_hgnc,
        'ensembl_to_uniprot': ensembl_to_uniprot,
        'symbol_to_id': symbol_to_id,
        'ot_ensembls': ot_ensembls
    }

def export_mapping(mapping_dict, out_path_base):
    for key, val in mapping_dict.items():
        f = f"{out_path_base}.{key}.json"
        print(f"Exporting {key} mapping to {f}")
        with open(f, 'w') as fp:
            json.dump(val, fp, indent=2)

if __name__ == "__main__":
    # Paths: update as needed
    DRUGBANK_PICKLE = "path/to/drugbank_parsed.pickle"
    OT_DRUGS_PARQUET = "path/to/open_targets_drugs.parquet"
    MESH_PICKLE = "path/to/mesh_data_2025.pickle"
    OT_DISEASES_PARQUET = "/Users/vi/Documents/not_work/drug_disease_interaction/data/processed/open_targets_merged/disease.parquet"
    # And so on

    # Drug mappings
    drugmaps = build_drug_mappings(DRUGBANK_PICKLE, OT_DRUGS_PARQUET)
    export_mapping(drugmaps, "out/drug_mappings")

    # Disease mappings
    diseasemaps = build_disease_mappings(MESH_PICKLE, OT_DISEASES_PARQUET)
    export_mapping(diseasemaps, "out/disease_mappings")

    # Target mappings (example, update paths as needed)
    # targetmaps = build_target_mappings(TARGET_PICKLE, OT_TARGETS_PARQUET)
    # export_mapping(targetmaps, "out/target_mappings")

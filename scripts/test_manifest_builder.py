import glob, os, re, json
import pandas as pd
import numpy as np

# Load paper text
with open('docs/IO-VNBD_paper_extracted.txt', 'r', encoding='utf-8') as f:
    paper_text = f.read()

# First, collect all files from disk
all_csvs = glob.glob('data/raw/IO-VNBD/**/*.csv', recursive=True)

# Organize files by rec_id and type
# Notice filenames can be: V-S1.csv, S-S1.csv, V-vta1a.csv, S-Vta1a.csv, V-Vfb01a.csv, etc.
# Normalize ID: remove V-, S-, v-, s-, .csv
def normalize_id(fname):
    base = os.path.splitext(fname)[0]
    if base.startswith(('V-', 'S-', 'v-', 's-', 'V_', 'S_')):
        return base[2:].upper()
    return base.upper()

# Map files
v_files_map = {}
s_files_map = {}

for f in all_csvs:
    rel = os.path.relpath(f, 'data/raw/IO-VNBD')
    fname = os.path.basename(f)
    nid = normalize_id(fname)
    is_sync = 'Synchronised' in rel
    
    if fname.upper().startswith(('V-', 'V_')) or (not fname.upper().startswith(('S-', 'S_')) and fname.upper().startswith('V')):
        if nid not in v_files_map or (is_sync and not v_files_map[nid]['is_sync']):
            v_files_map[nid] = {'path': rel, 'fname': fname, 'is_sync': is_sync, 'full_path': f}
    elif fname.upper().startswith(('S-', 'S_')):
        if nid not in s_files_map or (is_sync and not s_files_map[nid]['is_sync']):
            s_files_map[nid] = {'path': rel, 'fname': fname, 'is_sync': is_sync, 'full_path': f}

all_ids = sorted(list(set(v_files_map.keys()) | set(s_files_map.keys())))
print(f'Total unique recording IDs found on disk: {len(all_ids)}')
print(f'V recordings count: {len(v_files_map)}')
print(f'S recordings count: {len(s_files_map)}')
print(f'Paired (both V and S on disk): {len(set(v_files_map.keys()) & set(s_files_map.keys()))}')

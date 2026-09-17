import glob, os, re
import pandas as pd
import numpy as np

all_csvs = glob.glob('data/raw/IO-VNBD/**/*.csv', recursive=True)
print(f'Total CSV files: {len(all_csvs)}')

file_records = []
for f in all_csvs:
    rel_path = os.path.relpath(f, 'data/raw/IO-VNBD')
    fname = os.path.basename(f)
    size_bytes = os.path.getsize(f)
    is_sync = 'Synchronised' in rel_path
    
    # Determine type and ID
    # e.g. V-S1.csv, S-S1.csv, V-vta1a.csv, S-Vta1a.csv, etc.
    prefix = None
    rec_id = None
    if fname.upper().startswith('V-') or fname.upper().startswith('V_'):
        prefix = 'V'
        rec_id = fname[2:-4]
    elif fname.upper().startswith('S-') or fname.upper().startswith('S_'):
        prefix = 'S'
        rec_id = fname[2:-4]
    else:
        # Check without hyphen: e.g. Vta1.csv
        prefix = fname[0].upper()
        rec_id = fname[1:-4]
    
    file_records.append({
        'rel_path': rel_path,
        'fname': fname,
        'prefix': prefix,
        'rec_id': rec_id.upper() if rec_id else None,
        'size_bytes': size_bytes,
        'is_sync': is_sync
    })

df_files = pd.DataFrame(file_records)
print(df_files['prefix'].value_counts())
print('\nSync vs Unsync:')
print(df_files.groupby(['is_sync', 'prefix']).size())
print('\nTotal unique rec_ids in V:', df_files[df_files['prefix']=='V']['rec_id'].nunique())
print('Total unique rec_ids in S:', df_files[df_files['prefix']=='S']['rec_id'].nunique())

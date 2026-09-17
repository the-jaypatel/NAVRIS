import glob, os
import pandas as pd

all_csvs = glob.glob('data/raw/IO-VNBD/**/*.csv', recursive=True)
print(f'Checking {len(all_csvs)} CSV files for structure / anomalies...')

anomalies = []
for f in all_csvs:
    rel = os.path.relpath(f, 'data/raw/IO-VNBD')
    try:
        with open(f, 'r', encoding='latin1') as fp:
            header_line = fp.readline().strip()
            first_row = fp.readline().strip()
            
        header_cols = header_line.split(',')
        row_cols = first_row.split(',')
        
        # Check trailing empty
        if header_cols and header_cols[-1] == '':
            header_cols = header_cols[:-1]
        if row_cols and row_cols[-1] == '':
            row_cols = row_cols[:-1]
            
        if len(header_cols) != len(row_cols):
            anomalies.append({
                'file': rel,
                'issue': f'Header col count ({len(header_cols)}) != Row col count ({len(row_cols)})',
                'header': header_cols[:10],
                'row': row_cols[:10]
            })
            continue
            
        # Check if Date column contains actual date or integer
        df_sample = pd.read_csv(f, encoding='latin1', nrows=5)
        cols = [c.strip() for c in df_sample.columns]
        date_cols = [c for c in cols if 'date' in c.lower()]
        if date_cols:
            col_name = [c for c in df_sample.columns if 'date' in c.lower()][0]
            val = str(df_sample[col_name].iloc[0])
            if '-' not in val and ':' not in val:
                anomalies.append({
                    'file': rel,
                    'issue': f'Date column does not look like date: {val} (possible column shift)'
                })
    except Exception as e:
        anomalies.append({
            'file': rel,
            'issue': f'Exception reading file: {e}'
        })

print(f'Total anomalies found: {len(anomalies)}')
for a in anomalies:
    print(a['file'], '-->', a['issue'])

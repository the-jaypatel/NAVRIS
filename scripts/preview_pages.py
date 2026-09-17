with open('docs/IO-VNBD_paper_extracted.txt', 'r', encoding='utf-8') as f:
    text = f.read()
pages = text.split('=== PAGE ')
for p in pages[7:]: # from page 7 onwards
    lines = p.strip().split('\n')
    header = lines[0] if lines else ''
    print(f'=== PAGE {header} === (lines: {len(lines)})')
    for l in lines[1:15]:
        print('  ', l)

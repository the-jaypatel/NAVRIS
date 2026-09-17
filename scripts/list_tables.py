with open('docs/IO-VNBD_paper_extracted.txt', 'r', encoding='utf-8') as f:
    text = f.read()

import re
tables = re.findall(r'(Table A\d.*)', text)
for t in tables:
    print(t[:100])

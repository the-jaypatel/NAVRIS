import re

with open('docs/IO-VNBD_paper_extracted.txt', 'r', encoding='utf-8') as f:
    full_text = f.read()

# Let's see all dataset names mentioned in the paper:
# Typically lines start with Driver letter or dataset name: V-S1, V-S2, V-M, V-St1, V-Y1, V-Vta..., V-Vtb..., V-Vw..., V-Vfb..., S-T..., S-I, S-A...
matches = re.findall(r'(V-[A-Za-z0-9_]+|S-[A-Za-z0-9_]+)', full_text)
print('Total matches of V-*/S-* in paper:', len(matches))
from collections import Counter
c = Counter(matches)
print('Unique dataset names mentioned:', len(c))
print('Sample counts:', list(c.items())[:20])

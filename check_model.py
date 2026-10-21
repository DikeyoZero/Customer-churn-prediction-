import json

def show_nb(path, name):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f'=== {name} ({len(nb["cells"])} o) ===')
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])[:90].replace('\n', ' ')
        print(f'  [{i:02d}] {c["cell_type"]:8} | {src}')
    print()

show_nb(r'C:\Users\HP\Data Science\notebooks\03_model.ipynb', '03_model')

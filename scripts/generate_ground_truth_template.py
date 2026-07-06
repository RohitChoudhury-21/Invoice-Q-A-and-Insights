import pandas as pd
from pathlib import Path
files = sorted(Path('data/invoices_corpus').glob('*.pdf'))[:20]
df = pd.DataFrame({'invoice_number':'', 'vendor':'', 'invoice_date':'', 'total_amount':'', 'currency':'', 'line_items':'', 'source_file':[f.name for f in files]})
df.to_csv('data/ground_truth_template.csv', index=False)
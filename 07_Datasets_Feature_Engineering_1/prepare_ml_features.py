"""Export headerless, numeric-only 202-feature matrices from the master fish datasets."""
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent / 'fleeing_master'
FEATURES = ('Change_Area', 'DCenter_X', 'DCenter_Y', 'Vector_X', 'Vector_Y')
COLUMNS = [f'ID{fish}_F{frame}_{feature}' for fish in range(1, 5)
           for frame in range(10) for feature in FEATURES]
COLUMNS += ['Video_Avg_Pairwise_Distance', 'Video_Std_Pairwise_Distance']
assert len(COLUMNS) == 202


def export(source, destination):
    with source.open(newline='', encoding='utf-8-sig') as infile, destination.open('w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        missing = [c for c in COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f'Missing columns in {source}: {missing}')
        writer = csv.writer(outfile)
        count = 0
        for row in reader:
            values = [row[c] for c in COLUMNS]
            if any(v == '' or v is None for v in values):
                raise ValueError(f'Empty feature value in source row {count + 1}')
            for value in values:
                float(value)
            writer.writerow(values)
            count += 1
    return count


if __name__ == '__main__':
    for suffix in ('non_normalized', 'normalized'):
        source = BASE / f'master_fleeing_{suffix}.csv'
        destination = BASE / f'fleeing_ml_{suffix}.csv'
        print(f'{destination}: {export(source, destination)} rows, 202 columns')

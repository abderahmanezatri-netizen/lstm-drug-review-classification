import csv
import hashlib
import html
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'
LABELS = ['negative', 'neutral', 'positive']


def normalize(text):
    return ' '.join(html.unescape(text).lower().split())


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", normalize(text))


def load_raw(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'drugs.zip'
    if not archive.exists():
        temporary = directory / 'drugs.zip.part'
        urllib.request.urlretrieve(URL, temporary)
        # A failed download must not leave a corrupt file treated as a cache hit.
        with zipfile.ZipFile(temporary) as candidate:
            if candidate.testzip() is not None:
                raise ValueError('Corrupt UCI archive')
        temporary.replace(archive)
    splits, seen = {}, set()
    # Train wins collisions; official test duplicates are excluded.
    with zipfile.ZipFile(archive) as z:
        for split, suffix in [('train', 'drugsComTrain_raw.tsv'), ('test', 'drugsComTest_raw.tsv')]:
            member = next(n for n in z.namelist() if n.endswith(suffix))
            rows, removed = [], 0
            with z.open(member) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'), delimiter='\t'):
                    text = normalize(row['review'])
                    key = hashlib.sha256(text.encode()).hexdigest()
                    if not text or key in seen:
                        removed += 1
                        continue
                    seen.add(key)
                    rating = float(row['rating'])
                    label = 0 if rating <= 4 else (1 if rating <= 6 else 2)
                    rows.append({'id': row.get('', key), 'key': key, 'text': text, 'label': label})
            splits[split] = rows
            splits[split + '_removed'] = removed
    splits['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    return splits


def subset(rows, size, seed):
    if not size or size >= len(rows):
        return rows
    selected, _ = train_test_split(rows, train_size=size, stratify=[r['label'] for r in rows], random_state=seed)
    return selected


def prepare(directory, output, train_size, val_size, test_size, seed, vocab_size):
    raw = load_raw(directory)
    train, val = train_test_split(raw['train'], test_size=0.15, random_state=seed,
                                 stratify=[r['label'] for r in raw['train']])
    splits = {'train': subset(train, train_size, seed), 'val': subset(val, val_size, seed),
              'test': subset(raw['test'], test_size, seed)}
    counts = Counter(t for row in splits['train'] for t in tokenize(row['text']))
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({word: i + 2 for i, (word, _) in enumerate(counts.most_common(vocab_size - 2))})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    manifest = {k: [r['key'] for r in rows] for k, rows in splits.items()}
    (output / 'split_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    stats = {'archive_sha256': raw['archive_sha256'], 'duplicates_removed': {
        k: raw[k + '_removed'] for k in ['train', 'test']}, 'vocab_size': len(vocab)}
    for k, rows in splits.items():
        lengths = sorted(len(tokenize(r['text'])) for r in rows)
        stats[k] = {'size': len(rows), 'classes': dict(Counter(LABELS[r['label']] for r in rows)),
                    'length_median': lengths[len(lengths)//2], 'length_p90': lengths[int(len(lengths)*.9)]}
    (output / 'eda.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return splits, vocab, stats


class Reviews(Dataset):
    def __init__(self, rows, vocab, max_length):
        self.items = [(torch.tensor([vocab.get(w, 1) for w in tokenize(r['text'])[:max_length]] or [1]), r['label']) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    tokens, labels = zip(*batch)
    return pad_sequence(tokens, batch_first=True), torch.tensor([len(t) for t in tokens]), torch.tensor(labels)

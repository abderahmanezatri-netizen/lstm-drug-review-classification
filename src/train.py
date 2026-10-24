"""Run with python -m src.train; test is opened only after validation selection."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from .data import prepare, Reviews, collate, LABELS
from .models import Classifier


def evaluate(model, loader, device):
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x.to(device), lengths.to(device))
            truth.extend(y.tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
    return {'accuracy': accuracy_score(truth, predictions),
            'macro_f1': f1_score(truth, predictions, labels=[0, 1, 2], average='macro', zero_division=0),
            'report': classification_report(truth, predictions, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0),
            'confusion': confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist()}


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output)
    splits, vocab, stats = prepare(args.data, output, args.train_size, args.val_size, args.test_size, args.seed, args.vocab_size)
    datasets = {k: Reviews(rows, vocab, args.max_length) for k, rows in splits.items()}
    val_loader = DataLoader(datasets['val'], batch_size=128, collate_fn=collate)
    test_loader = DataLoader(datasets['test'], batch_size=128, collate_fn=collate)
    configs = [('reference', .001, 'adam', 64), ('learning_rate', .003, 'adam', 64),
               ('optimizer', .001, 'rmsprop', 64), ('batch_size', .001, 'adam', 32)]
    if args.smoke:
        configs = configs[:1]
    environment = {'torch': torch.__version__, 'device': str(device), 'args': vars(args),
                   'numpy': np.__version__, 'classes': LABELS, 'configuration_grid': configs}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    all_runs, selected = [], []
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    for variant in ['vanilla', 'cifg', 'wmc']:
        best_variant, best_path = None, None
        for name, lr, optimizer_name, batch in configs:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(args.seed)
            model = Classifier(len(vocab), variant, args.embedding_dim, args.hidden_size).to(device)
            optimizer = (torch.optim.Adam if optimizer_name == 'adam' else torch.optim.RMSprop)(model.parameters(), lr=lr)
            # Inverse-frequency weights are calculated on TRAIN ONLY.
            counts = np.bincount([r['label'] for r in splits['train']], minlength=3)
            weights = torch.tensor(len(datasets['train']) / (3 * counts), dtype=torch.float32, device=device)
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
            generator = torch.Generator().manual_seed(args.seed)
            loader = DataLoader(datasets['train'], batch_size=batch, shuffle=True, generator=generator, collate_fn=collate)
            history, best_f1, start = [], -1., time.perf_counter()
            path = checkpoints / f'{variant}_{name}.pt'
            for epoch in range(args.epochs):
                model.train()
                loss_sum = 0.
                for x, lengths, y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x.to(device), lengths.to(device)), y.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    loss_sum += loss.item() * y.size(0)
                metrics = evaluate(model, val_loader, device)
                history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(datasets['train']),
                                'val_macro_f1': metrics['macro_f1'], 'val_accuracy': metrics['accuracy']})
                print(variant, name, history[-1], flush=True)
                if metrics['macro_f1'] > best_f1:
                    best_f1 = metrics['macro_f1']
                    torch.save({'state_dict': model.state_dict(), 'variant': variant,
                                'embedding_dim': args.embedding_dim, 'hidden_size': args.hidden_size,
                                'vocab_size': len(vocab), 'max_length': args.max_length}, path)
            result = {'variant': variant, 'configuration': name, 'lr': lr, 'optimizer': optimizer_name,
                      'batch_size': batch, 'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'val_macro_f1': best_f1, 'seconds': time.perf_counter() - start, 'history': history}
            all_runs.append(result)
            (output / 'runs.json').write_text(json.dumps(all_runs, indent=2), encoding='utf-8')
            if best_variant is None or best_f1 > best_variant['val_macro_f1']:
                best_variant, best_path = result, path
        selected.append((best_variant, best_path))
    # Configurations and epochs are now locked. Never select using test scores.
    final = []
    for result, path in selected:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = Classifier(len(vocab), result['variant'], args.embedding_dim, args.hidden_size).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        final.append({**{k: v for k, v in result.items() if k != 'history'},
                      'checkpoint': path.as_posix(), 'test': evaluate(model, test_loader, device)})
    majority = int(np.argmax(np.bincount([r['label'] for r in splits['train']], minlength=3)))
    truth = [r['label'] for r in splits['test']]
    baseline = {'accuracy': accuracy_score(truth, [majority]*len(truth)),
                'macro_f1': f1_score(truth, [majority]*len(truth), labels=[0,1,2], average='macro', zero_division=0)}
    (output / 'results.json').write_text(json.dumps({'models': final, 'majority_baseline': baseline}, indent=2), encoding='utf-8')
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'configuration', 'parameters', 'val_macro_f1', 'test_macro_f1', 'test_accuracy'])
        writer.writeheader()
        for r in final:
            writer.writerow({k: r[k] for k in ['variant','configuration','parameters','val_macro_f1']} |
                            {'test_macro_f1': r['test']['macro_f1'], 'test_accuracy': r['test']['accuracy']})
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name, default in [('train-size', 3000), ('val-size', 750), ('test-size', 1000), ('epochs', 3),
                          ('seed', 42), ('vocab-size', 12000), ('max-length', 120),
                          ('embedding-dim', 64), ('hidden-size', 48), ('threads', 2)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='results/local')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())

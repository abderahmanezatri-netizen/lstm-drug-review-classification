"""Time-budgeted comparison of the three manually implemented LSTM cells.

Every completed round trains ALL 12 settings for one epoch. Only whole rounds
are committed, so a deadline cannot give the last architecture fewer epochs.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from .data import prepare, Reviews, collate, LABELS
from .models import Classifier
from .train import evaluate


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def benchmark(device, batch_size, args):
    """Synthetic timing only; no validation or test labels are involved."""
    seed_everything(args.seed)
    model = Classifier(2000, 'wmc', args.embedding_dim, args.hidden_size,
                       pooling='mean', forget_bias=1.).to(device)
    x = torch.randint(1, 2000, (batch_size, args.max_length), device=device)
    lengths = torch.full((batch_size,), args.max_length, device=device)
    y = torch.arange(batch_size, device=device) % 3
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    durations = []
    for _ in range(3):
        synchronize(device)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x, lengths), y)
        loss.backward()
        optimizer.step()
        synchronize(device)
        durations.append(time.monotonic() - started)
    return float(np.median(durations[1:]))


def run(args):
    started = time.monotonic()
    deadline = started + args.minutes * 60
    # Keep time for restoring checkpoints and evaluating the three final models.
    training_deadline = deadline - min(90., args.minutes * 60 * .15)
    output = Path(args.output)
    if (output / 'runs.json').exists():
        raise ValueError('This output already contains experiments. Choose a new --output directory.')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Choose --device cpu or install a CUDA-enabled PyTorch build.')
    timings = {}
    if args.device == 'auto':
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for name in devices:
            timings[name] = benchmark(torch.device(name), args.batch_size, args)
            print(f'{name}: {timings[name]:.3f} seconds/training batch', flush=True)
        device = torch.device(min(timings, key=timings.get))
    else:
        device = torch.device(args.device)
    print(f'Device: {device}; target total budget: {args.minutes:g} minutes', flush=True)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0), flush=True)

    splits, vocab, stats = prepare(args.data, output, args.train_size, args.val_size,
                                  args.test_size, args.seed, args.vocab_size)
    datasets = {key: Reviews(rows, vocab, args.max_length) for key, rows in splits.items()}
    pin = device.type == 'cuda'
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size,
                            collate_fn=collate, pin_memory=pin)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size,
                             collate_fn=collate, pin_memory=pin)
    counts = np.bincount([row['label'] for row in splits['train']], minlength=3)
    # Four settings per architecture: learning rate, batch size, loss weighting.
    # Each alternative differs from the reference in exactly one setting.
    configs = [('reference', .003, args.batch_size, 0.),
               ('learning_rate', .001, args.batch_size, 0.),
               ('batch_size', .003, max(16, args.batch_size // 2), 0.),
               ('class_weight', .003, args.batch_size, .5)]
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    vocab_hash = hashlib.sha256((output / 'vocab.json').read_bytes()).hexdigest()
    environment = {'torch': torch.__version__, 'device': str(device),
                   'gpu': torch.cuda.get_device_name(0) if pin else None,
                   'args': vars(args), 'numpy': np.__version__, 'classes': LABELS,
                   'benchmark_seconds_per_batch': timings, 'configuration_grid': configs,
                   'pooling': 'masked mean of recurrent hidden states', 'forget_bias': 1.,
                   'selection': 'macro F1 on validation; test evaluated only after selection',
                   'budget_policy': 'equal complete rounds; unfinished round excluded from selection'}
    save_json(output / 'environment.json', environment)
    states = []
    for variant in ['vanilla', 'cifg', 'wmc']:
        for name, lr, batch, power in configs:
            seed_everything(args.seed)
            model = Classifier(len(vocab), variant, args.embedding_dim, args.hidden_size,
                               pooling='mean', forget_bias=1.).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            weights = torch.tensor((len(datasets['train']) / (3 * counts)) ** power,
                                   dtype=torch.float32, device=device)
            result = {'variant': variant, 'configuration': name, 'lr': lr, 'optimizer': 'adam',
                      'batch_size': batch, 'class_weight_power': power,
                      'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'val_macro_f1': -1., 'best_epoch': 0, 'seconds': 0., 'history': []}
            path = checkpoints / f'{variant}_{name}.pt'
            states.append({'model': model, 'optimizer': optimizer,
                           'criterion': torch.nn.CrossEntropyLoss(weight=weights),
                           'result': result, 'path': path})

    complete_rounds, round_times, stop_reason = 0, [], 'epoch_limit'
    for epoch in range(args.epochs):
        if round_times and time.monotonic() + max(round_times[-2:]) * 1.3 > training_deadline:
            stop_reason = 'insufficient_time_for_another_complete_round'
            break
        round_started = time.monotonic()
        pending, interrupted = [], False
        for state in states:
            model, optimizer, result = state['model'], state['optimizer'], state['result']
            # Identical epoch seed/order for matching batch sizes in all architectures.
            seed_everything(args.seed + epoch)
            loader = DataLoader(datasets['train'], batch_size=result['batch_size'], shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed + epoch),
                                collate_fn=collate, pin_memory=pin)
            run_started = time.monotonic()
            model.train()
            loss_sum = torch.zeros((), device=device)
            for x, lengths, y in loader:
                if time.monotonic() >= training_deadline:
                    interrupted = True
                    break
                optimizer.zero_grad(set_to_none=True)
                logits = model(x.to(device, non_blocking=pin), lengths.to(device, non_blocking=pin))
                loss = state['criterion'](logits, y.to(device, non_blocking=pin))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                loss_sum += loss.detach() * y.size(0)
            if interrupted:
                break
            metrics = evaluate(model, val_loader, device)
            history = {'epoch': epoch + 1, 'train_loss': loss_sum.item() / len(datasets['train']),
                       'val_macro_f1': metrics['macro_f1'], 'val_accuracy': metrics['accuracy']}
            seconds = time.monotonic() - run_started
            improved = metrics['macro_f1'] > result['val_macro_f1']
            if improved:
                torch.save({'state_dict': model.state_dict(), 'variant': result['variant'],
                            'embedding_dim': args.embedding_dim, 'hidden_size': args.hidden_size,
                            'vocab_size': len(vocab), 'max_length': args.max_length,
                            'pooling': 'mean', 'vocab_sha256': vocab_hash},
                           state['path'].with_suffix('.pending'))
            pending.append((state, history, seconds, improved))
            print(f"Epoch {epoch + 1}/{args.epochs} | {result['variant']} {result['configuration']} | "
                  f"val accuracy {metrics['accuracy']:.3f} | macro F1 {metrics['macro_f1']:.3f} | "
                  f"{seconds:.1f}s", flush=True)
        if interrupted:
            stop_reason = 'deadline_during_round_previous_complete_round_retained'
            break
        # All twelve models completed the epoch: commit the complete round atomically per run.
        for state, history, seconds, improved in pending:
            result = state['result']
            result['history'].append(history)
            result['seconds'] += seconds
            if improved:
                state['path'].with_suffix('.pending').replace(state['path'])
                result['val_macro_f1'], result['best_epoch'] = history['val_macro_f1'], epoch + 1
        complete_rounds += 1
        round_times.append(time.monotonic() - round_started)
        save_json(output / 'runs.json', [state['result'] for state in states])
        save_json(output / 'progress.json', {'complete_epochs_per_run': complete_rounds,
                  'elapsed_seconds': time.monotonic() - started, 'last_round_seconds': round_times[-1],
                  'target_seconds': args.minutes * 60})
        print(f'Completed round {complete_rounds}: {round_times[-1]:.1f}s; '
              f'elapsed {(time.monotonic()-started)/60:.1f} min', flush=True)

    for state in states:
        state['path'].with_suffix('.pending').unlink(missing_ok=True)
    if complete_rounds == 0:
        save_json(output / 'budget.json', {'status': 'incomplete_no_comparable_round',
                  'stop_reason': stop_reason, 'seconds': time.monotonic() - started})
        raise RuntimeError('Budget too short for one complete comparison. Reduce --train-size or increase --minutes.')

    # Lock all choices on validation BEFORE opening the test evaluation.
    selected = []
    for variant in ['vanilla', 'cifg', 'wmc']:
        selected.append(max((s for s in states if s['result']['variant'] == variant),
                            key=lambda s: s['result']['val_macro_f1']))
    final_choice = max(selected, key=lambda s: s['result']['val_macro_f1'])['result']['variant']
    # Reload the committed checkpoints; an interrupted round cannot affect evaluation.
    final = []
    for state in selected:
        result = state['result']
        checkpoint = torch.load(state['path'], map_location=device, weights_only=True)
        model = Classifier(len(vocab), result['variant'], args.embedding_dim,
                           args.hidden_size, pooling='mean').to(device)
        model.load_state_dict(checkpoint['state_dict'])
        final.append({**{k: v for k, v in result.items() if k != 'history'},
                      'checkpoint': state['path'].as_posix(),
                      'test': evaluate(model, test_loader, device)})
    majority = int(np.argmax(counts))
    truth = [row['label'] for row in splits['test']]
    predictions = [majority] * len(truth)
    baseline = {'accuracy': accuracy_score(truth, predictions),
                'macro_f1': f1_score(truth, predictions, labels=[0, 1, 2], average='macro', zero_division=0)}
    save_json(output / 'results.json', {'models': final, 'majority_baseline': baseline,
              'final_model_selected_on_validation': final_choice, 'complete_epochs_per_run': complete_rounds})
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'configuration', 'parameters',
                  'best_epoch', 'val_macro_f1', 'test_macro_f1', 'test_accuracy'])
        writer.writeheader()
        for result in final:
            writer.writerow({k: result[k] for k in ['variant', 'configuration', 'parameters', 'best_epoch', 'val_macro_f1']}
                            | {'test_macro_f1': result['test']['macro_f1'], 'test_accuracy': result['test']['accuracy']})
    budget = {'status': 'complete', 'stop_reason': stop_reason,
              'complete_epochs_per_run': complete_rounds, 'number_of_runs': len(states),
              'total_seconds': time.monotonic() - started, 'target_seconds': args.minutes * 60,
              'round_seconds': round_times}
    save_json(output / 'budget.json', budget)
    print(json.dumps({'budget': budget, 'models': [{k: m[k] for k in ['variant', 'configuration', 'test']} for m in final]}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('train-size', 15000), ('val-size', 1500), ('test-size', 3000),
                          ('epochs', 12), ('seed', 42), ('vocab-size', 18000),
                          ('max-length', 160), ('embedding-dim', 64), ('hidden-size', 64),
                          ('batch-size', 256), ('threads', 4)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--minutes', type=float, default=45.)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='results/fast')
    args = parser.parse_args()
    if args.minutes <= 0 or args.epochs < 1 or args.batch_size < 16:
        parser.error('Use positive minutes, at least one epoch, and batch size >=16.')
    run(args)

"""Export the measured GPU run, its demo commands, and a self-contained ZIP."""
import argparse
import csv
import json
import zipfile
from pathlib import Path


def export(run_directory, destination):
    root = Path(__file__).resolve().parents[1]
    run_directory = Path(run_directory)
    if not run_directory.is_absolute():
        run_directory = root / run_directory
    results = json.loads((run_directory / 'results.json').read_text(encoding='utf-8'))
    budget = json.loads((run_directory / 'budget.json').read_text(encoding='utf-8'))
    runs = json.loads((run_directory / 'runs.json').read_text(encoding='utf-8'))
    assert len(runs) == 12
    assert len({len(r['history']) for r in runs}) == 1
    assert all(r['history'] for r in runs)
    selected = results['models']
    for model in selected:
        assert (root / model['checkpoint']).is_file(), model['checkpoint']
    destination = Path(destination)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    fields = ['variant', 'configuration', 'lr', 'optimizer', 'batch_size',
              'class_weight_power', 'parameters', 'val_macro_f1', 'best_epoch', 'seconds']
    with (destination / 'hyperparameters.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ['completed_epochs'])
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run[key] for key in fields}
                            | {'completed_epochs': len(run['history'])})
    table = ['MODEL | CONFIGURATION | VALIDATION MACRO F1 | TEST ACCURACY | TEST MACRO F1']
    for model in selected:
        table.append(f"{model['variant'].upper()} | {model['configuration']} | "
                     f"{model['val_macro_f1']:.4f} | {model['test']['accuracy']:.2%} | {model['test']['macro_f1']:.4f}")
    summary = ('GPU EXPERIMENT - ACTUAL MEASURED RESULTS\n\n'
               f"Duration: {budget['total_seconds']/60:.2f} minutes\n"
               f"Complete epochs per setting: {budget['complete_epochs_per_run']}\n"
               f"Number of settings: {budget['number_of_runs']}\n"
               f"Stop reason: {budget['stop_reason']}\n"
               f"Final model chosen on validation: {results['final_model_selected_on_validation'].upper()}\n\n"
               + '\n'.join(table) + '\n\n'
               f"Majority baseline: accuracy {results['majority_baseline']['accuracy']:.2%}, "
               f"macro F1 {results['majority_baseline']['macro_f1']:.4f}\n\n"
               'The same stratified splits and completed training epochs are used for all cells.\n'
               'The test is evaluated after validation-based selection. These are not full-dataset results.\n'
               'See ENTRAINEMENT_RAPIDE.md and the run environment.json for the exact protocol.\n'
               'The final report in output/pdf/report.pdf describes this GPU study.\n'
               'Slides are intentionally left for later. Fill report_details.json before submission.\n')
    (destination / 'RESULTATS_GPU.txt').write_text(summary, encoding='utf-8')

    command_file = ('DEMO - NEW GPU-TRAINED MODELS\n\n'
                    'On the original computer, open PowerShell in:\n'
                    'C:\\Users\\abder\\Desktop\\WORK\\tp\\nlp\\projet_rnn\n\n'
                    'The commands below load the three validation-selected checkpoints.\n'
                    'No retraining is needed. Use the same review to compare all three.\n\n')
    for model in selected:
        command_file += (f"{model['variant'].upper()} ({model['configuration']}):\n"
                         f"..\\.venv\\python.exe -m src.predict --checkpoint {model['checkpoint']} "
                         '--text "This medicine helped me sleep but caused nausea."\n\n')
    command_file += ('The vocabulary is found automatically beside the checkpoints directory.\n'
                     'Use the three selected checkpoints in results/gpu_45min/checkpoints.\n\n'
                     'To rerun the same completed epoch count in a NEW output folder:\n'
                     f"..\\.venv\\python.exe -m src.train_fast --device cuda --minutes 45 "
                     f"--epochs {budget['complete_epochs_per_run']} --output results/my_gpu_run\n"
                     'On slower hardware the time budget can still reduce the epoch count; check budget.json.\n')
    (destination / 'COMMANDES_DEMO_GPU.txt').write_text(command_file, encoding='utf-8')

    friend = ('INSTALL ON ANOTHER WINDOWS LAPTOP\n\n'
              '1. Extract this ZIP, then open a terminal in the extracted folder containing src/.\n'
              '2. Use Python 3.12 to create a new environment:\n\n'
              'py -3.12 -m venv .venv\n'
              '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n\n'
              '3. Run the implementation checks:\n\n'
              '.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v\n\n'
              '4. Test the saved models using the commands below.\n'
              'No GPU, training dataset, Colab or retraining is needed for prediction.\n'
              'Internet is needed for installing dependencies.\n\n')
    for model in selected:
        friend += (f"{model['variant'].upper()}:\n"
                   f".\\.venv\\Scripts\\python.exe -m src.predict --checkpoint {model['checkpoint']} "
                   '--text "This medicine helped me sleep but caused nausea."\n\n')
    friend += ('The original owner uses a Conda environment in the parent folder.\n'
               'Your new environment is a regular venv inside this extracted project.\n'
               'That explains the different Python paths.\n')
    (destination / 'INSTALLATION_AMI_GPU.txt').write_text(friend, encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figures = destination / 'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    names = [m['variant'].upper() for m in selected] + ['Majority']
    axes[0].bar(names, [m['test']['accuracy'] for m in selected]
                + [results['majority_baseline']['accuracy']], color='#147D82')
    axes[0].set(title='Held-out test accuracy', ylim=(0, 1), ylabel='Accuracy')
    axes[1].bar(names, [m['test']['macro_f1'] for m in selected]
                + [results['majority_baseline']['macro_f1']], color='#36658A')
    axes[1].set(title='Held-out test macro F1', ylim=(0, 1), ylabel='Macro F1')
    fig.tight_layout()
    fig.savefig(figures / 'comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, model in zip(axes, selected):
        matrix = np.array(model['test']['confusion'])
        ax.imshow(matrix, cmap='Blues')
        ax.set(title=model['variant'].upper(), xlabel='Predicted', ylabel='True')
        ax.set_xticks(range(3), ['Negative', 'Neutral', 'Positive'])
        ax.set_yticks(range(3), ['Negative', 'Neutral', 'Positive'])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.max()/2 else '#19334D')
    fig.tight_layout()
    fig.savefig(figures / 'confusion.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for model in selected:
        run = next(r for r in runs if r['variant'] == model['variant']
                   and r['configuration'] == model['configuration'])
        ax.plot([h['epoch'] for h in run['history']],
                [h['val_macro_f1'] for h in run['history']], marker='o',
                label=f"{model['variant'].upper()} ({model['configuration']})")
    ax.set(title='Validation macro F1: selected configurations', xlabel='Completed epoch', ylabel='Macro F1')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'validation_curves.png', dpi=180)
    plt.close(fig)
    archive_path = destination / 'Projet_RNN_GPU.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for directory in ['src', 'tests']:
            for path in (root / directory).glob('*.py'):
                archive.write(path, path.relative_to(root).as_posix())
        for name in ['requirements.txt', 'README.md', 'LIRE_MOI.md', 'ENTRAINEMENT_RAPIDE.md', 'report_details.json', '.gitignore']:
            if (root / name).is_file():
                archive.write(root / name, name)
        for name in ['export_gpu_project.py', 'build_report.py']:
            archive.write(root / 'tools' / name, 'tools/' + name)
        for name in ['report.pdf', 'report.md']:
            report = root / 'output/pdf' / name
            if report.exists():
                archive.write(report, 'output/pdf/' + name)
        for path in run_directory.glob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
        for model in selected:
            path = root / model['checkpoint']
            archive.write(path, path.relative_to(root).as_posix())
        for name in ['RESULTATS_GPU.txt', 'COMMANDES_DEMO_GPU.txt', 'INSTALLATION_AMI_GPU.txt', 'hyperparameters.csv']:
            archive.write(destination / name, name)
        for path in figures.glob('*.png'):
            archive.write(path, 'output/gpu/figures/' + path.name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.endswith('.pt')]) == 3
    print(summary)
    print(f'Archive: {archive_path} ({archive_path.stat().st_size/1024**2:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='results/gpu_45min')
    parser.add_argument('--output', default='output/gpu')
    args = parser.parse_args()
    export(args.run, args.output)

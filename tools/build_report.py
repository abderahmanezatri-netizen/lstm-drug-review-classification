"""Build the final five-page report from the recorded GPU experiment."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak


def build(run='results/gpu_45min', output='output'):
    root = Path(__file__).resolve().parents[1]
    base, out = root / run, root / output
    pdf_dir = out / 'pdf'
    pdf_dir.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((base / name).read_text(encoding='utf-8'))
    data, results, runs, env, budget = [read(name) for name in
        ['eda.json', 'results.json', 'runs.json', 'environment.json', 'budget.json']]
    models, args = results['models'], env['args']
    details_path = root / 'report_details.json'
    details = json.loads(details_path.read_text(encoding='utf-8')) if details_path.exists() else {}
    members = ', '.join(name for name in details.get('team_members', []) if name.strip())
    epochs, duration = budget['complete_epochs_per_run'], round(budget['total_seconds'])
    classes = ['negative', 'neutral', 'positive']
    figures = out / 'gpu/figures'
    for name in ['validation_curves.png', 'confusion.png']:
        if not (figures / name).is_file():
            raise FileNotFoundError('Run tools/export_gpu_project.py before building the report.')
    styles = {
        'body': ParagraphStyle('body', fontName='Times-Roman', fontSize=10.7, leading=14.1, spaceAfter=8),
        'h': ParagraphStyle('h', fontName='Times-Bold', fontSize=13.5, leading=17, spaceBefore=9, spaceAfter=7),
        'title': ParagraphStyle('title', fontName='Times-Bold', fontSize=21, leading=25, spaceAfter=12),
        'small': ParagraphStyle('small', fontName='Times-Roman', fontSize=9, leading=11.7, spaceAfter=6),
        'cell': ParagraphStyle('cell', fontName='Times-Roman', fontSize=10, leading=12.3),
    }
    story, markdown = [], []
    def text(value, style='body'):
        story.append(Paragraph(escape(value).replace('\n', '<br/>'), styles[style]))
        markdown.append(('# ' + value.replace('\n', ' ') if style == 'title' else value) + '\n')
    def heading(value):
        text(value, 'h')
        markdown[-1] = '## ' + value + '\n'
    def table(rows, widths):
        items = [[Paragraph(escape(str(c)).replace('\n', '<br/>'), styles['cell']) for c in row] for row in rows]
        obj = Table(items, colWidths=widths, hAlign='LEFT', repeatRows=1)
        obj.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECEFF1')),
            ('LINEBELOW', (0,0), (-1,0), .6, colors.HexColor('#56616B')),
            ('LINEBELOW', (0,-1), (-1,-1), .3, colors.HexColor('#9DA5AC')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.extend([obj, Spacer(1, 9)])
        markdown.append('| ' + ' | '.join(map(str, rows[0])) + ' |\n'
                        + '| ' + ' | '.join('---' for _ in rows[0]) + ' |\n'
                        + '\n'.join('| '+' | '.join(str(c).replace('\n','; ') for c in row)+' |' for row in rows[1:])+'\n')
    def figure(name, height, caption):
        story.append(Image(str(figures/name), width=491, height=height))
        markdown.append(f'![{caption}](../gpu/figures/{name})\n')
        text(caption, 'small')
    def page():
        story.append(PageBreak())
        markdown.append('\n---\n')

    text('Drug Review Classification\nwith Vanilla, CIFG and WMC LSTMs', 'title')
    text('Advanced Natural Language Processing | 2026/2027\n24 October 2026', 'small')
    identity = ['Team members: ' + (members or '__________________________________________________')]
    if details.get('lab_group', '').strip():
        identity.append('Lab group: ' + details['lab_group'])
    if details.get('repository_url', '').strip():
        identity.append('Public GitHub repository: ' + details['repository_url'])
    text('\n'.join(identity), 'small')
    heading('1. Project aim')
    text('Our task is to classify English medication reviews as negative, neutral or positive. We chose the Drug Review Dataset (Drugs.com), one of the options in the project guidelines [1]. We compare a vanilla LSTM with two cells from the appendix: Coupled Input-Forget Gate (CIFG) and Working Memory Connections (WMC). We also compare training settings to see whether they matter as much as the cell design.')
    text('The ratings provide the labels: 1-4 means negative, 5-6 neutral, and 7-10 positive. We chose these intervals for this project. The model receives only the written review. The rating, drug name, condition, date and useful-vote count are not input features.')
    heading('2. Dataset and preparation')
    text('The UCI archive [2] contains 161,297 training reviews and 53,766 test reviews in TSV format. The program reads them using a tab separator. It decodes HTML entities, converts text to lowercase and collapses repeated whitespace. After this normalization, exact duplicates are removed. The first occurrence is kept; training records take priority over identical test records.')
    text(f"The filter removes {data['duplicates_removed']['train']:,} records from the official training source and {data['duplicates_removed']['test']:,} from the test source. We reserve 15% of the cleaned training source for validation, then take stratified subsets with seed 42. This keeps the experiment within our time budget while preserving the class proportions.")
    table([['Partition', 'Reviews', 'Negative', 'Neutral', 'Positive']] +
          [[label, f"{data[key]['size']:,}"] + [f"{data[key]['classes'][c]:,}" for c in classes]
           for key, label in [('train','Training'), ('val','Validation'), ('test','Test')]], [111,80,100,100,100])
    text(f"Positive reviews make up {data['train']['classes']['positive']/data['train']['size']:.1%} of the training subset. The median review length is {data['train']['length_median']} tokens and the 90th percentile is {data['train']['length_p90']}. We therefore report macro F1 alongside accuracy and limit inputs to {args['max_length']} tokens. The vocabulary contains {data['vocab_size']:,} entries, including padding and unknown tokens, and is built from training text only.")
    page()

    heading('3. Models')
    text(f"All three classifiers use the same structure: a {args['embedding_dim']}-dimensional learned embedding, one recurrent layer with {args['hidden_size']} hidden units, masked mean pooling, dropout of 0.2 and a linear output layer with three logits. Mean pooling averages the hidden states over valid tokens. Padding positions are excluded, and padded steps cannot change the recurrent state.")
    text('The recurrent cells are written with nn.Linear rather than nn.LSTM. Their equations follow the course appendix [1]. For a gate q, let A_q = W_xq x_t + W_hq h_prev + b_q. Here sigmoid is the logistic function and * denotes elementwise multiplication. The candidate is g = tanh(A_g).')
    table([['Cell', 'Gate equations'],
           ['Vanilla', 'i = sigmoid(A_i)\nf = sigmoid(A_f)\no = sigmoid(A_o)'],
           ['CIFG', 'f = sigmoid(A_f)\ni = 1 - f\no = sigmoid(A_o)'],
           ['WMC', 'i = sigmoid(A_i + tanh(W_ci c_prev))\nf = sigmoid(A_f + tanh(W_cf c_prev))\no = sigmoid(A_o + tanh(W_co c_new))']], [81,410])
    text('For every cell, c_new = f * c_prev + i * g and h_new = o * tanh(c_new). In WMC, the input and forget gates read the previous memory, whereas the output gate reads the updated memory. CIFG removes an independent input gate, which reduces recurrent parameters. WMC adds three full memory-to-gate matrices. The readings listed in the handout are given in [3] and [4].')
    h, e = args['hidden_size'], args['embedding_dim']
    table([['Cell', 'Recurrent parameters', 'Total trainable parameters']] +
          [[m['variant'].upper(), f"{(3 if m['variant']=='cifg' else 4)*h*(e+h+1)+(3*h*h if m['variant']=='wmc' else 0):,}", f"{m['parameters']:,}"] for m in models], [81,190,220])
    text('The totals include the embeddings and output layer. Embeddings account for most parameters, so differences in recurrent complexity are less visible in the totals. We keep the hidden size fixed for all cells; the models are not matched to exactly the same parameter count.')
    text('Linear weights use Xavier uniform initialization. Biases start at zero except the forget-gate bias, which starts at 1. The embeddings use the PyTorch default initialization. We compute the input affine transform for all timesteps together, then apply the recurrent updates in sequence. This changes the computation order without changing the cell equations.')
    page()

    heading('4. Training and hyperparameter comparison')
    text('We use Adam and cross entropy, with gradient clipping at norm 1.0. Each cell is trained with four settings. Each alternative changes one setting relative to the reference. The weighted loss uses w_k = (N / (3 n_k))^0.5, where n_k is the number of training reviews in class k. Only training labels are used to calculate these weights.')
    table([['Setting', 'Learning rate', 'Batch size', 'Class weights'],
           ['Reference', '0.003', '256', 'None'], ['Learning rate', '0.001', '256', 'None'],
           ['Batch size', '0.003', '128', 'None'], ['Weighted loss', '0.003', '256', 'Square-root inverse frequency']], [123,100,88,180])
    text(f"This gives 12 runs in total. To keep the comparison balanced, the program trains every setting for one epoch before starting the next round. It completed {epochs} rounds on an {env['gpu']} using PyTorch {env['torch']}. The recorded total time was {duration//60} minutes {duration%60} seconds. A further round was not started because it could exceed the planned {args['minutes']:g}-minute budget after allowing time for evaluation.")
    text('For each run, we keep the epoch with the highest validation macro F1. We then choose one configuration per cell using that same metric. The final architecture is also selected on validation. Test scores are calculated after these choices are fixed. All runs share the same split manifest and seed. A smaller batch gives more updates per epoch, so this is an equal-epoch comparison, not an equal-update comparison.')
    table([['Best validation macro F1', 'Reference', 'LR 0.001', 'Batch 128', 'Weighted loss']] +
          [[variant.upper()] + [f"{next(r for r in runs if r['variant']==variant and r['configuration']==cfg)['val_macro_f1']:.4f}"
           for cfg in ['reference','learning_rate','batch_size','class_weight']] for variant in ['vanilla','cifg','wmc']], [139,88,88,88,88])
    text('The weighted loss gives the best validation macro F1 for each cell. The smaller learning rate gives lower best scores within this budget. Batch 128 improves on the reference for all three cells, but requires more updates. These observations apply to our tested settings; they do not show that one learning rate or weighting method is always preferable.')
    figure('validation_curves.png', 221, 'Figure 1. Validation macro F1 for the configuration selected for each cell. Later epochs do not always improve the score.')
    page()

    heading('5. Test results')
    text('The following scores use the same 3,000 held-out reviews. The majority reference always predicts positive, the most frequent class in training. Accuracy is the proportion of correct predictions. Macro F1 gives the same weight to the F1 score of each class, making errors on the smaller classes more visible.')
    table([['Model', 'Parameters', 'Best epoch', 'Val F1', 'Test accuracy', 'Test F1']] +
          [[m['variant'].upper(), f"{m['parameters']:,}", m['best_epoch'], f"{m['val_macro_f1']:.4f}", f"{m['test']['accuracy']:.2%}", f"{m['test']['macro_f1']:.4f}"] for m in models] +
          [['Majority', '0', '-', '-', f"{results['majority_baseline']['accuracy']:.2%}", f"{results['majority_baseline']['macro_f1']:.4f}"]], [68,88,65,80,95,95])
    text('CIFG has the highest observed test accuracy, while Vanilla has the highest test macro F1. WMC had the highest validation macro F1 and remains our validation-selected final model. We report the difference in rankings rather than changing the selection rule after looking at the test scores.')
    table([['Model', 'Negative F1', 'Neutral F1', 'Positive F1']] +
          [[m['variant'].upper()] + [f"{m['test']['report'][c]['f1-score']:.3f}" for c in classes] for m in models], [112,126,126,127])
    figure('confusion.png', 148, 'Figure 2. Test confusion matrices. Rows are true labels; columns are predicted labels. Every matrix contains 3,000 reviews.')
    text('Positive reviews are the easiest class for all three models. Neutral reviews are the main weakness. Vanilla correctly identifies 63 of 267 neutral reviews, CIFG 38 and WMC 32. Many neutral reviews are predicted as positive. A middle rating can correspond to mixed language, and the smaller number of neutral training examples may also contribute to these errors.')
    text('All three models exceed the majority reference on both metrics. Still, accuracy in the mid-seventies should not hide the low neutral-class recall. The confusion matrices are more informative than a single headline score.')
    page()

    heading('6. Discussion and limitations')
    text('WMC adds memory connections, but the test results do not show a clear advantage over Vanilla or CIFG. CIFG uses fewer recurrent parameters and achieves the highest test accuracy in this run. Vanilla better balances the three class F1 scores. The validation differences are small, and one seed is not enough to conclude that one cell is generally better.')
    text('The rating-to-label mapping is another limit. A review may mention serious side effects and still have a high rating, or describe both benefits and problems. Our label comes from the numeric rating rather than a human sentiment annotation. Keeping the first duplicate also retains one label when identical reviews have conflicting ratings. Exact deduplication does not remove paraphrases or guarantee separation by patient or drug.')
    text(f"We used a reduced training set and the first {args['max_length']} tokens of each review. Longer reviews can lose important information near the end. A larger dataset, several random seeds and a sequence-length comparison would make the conclusions stronger. We would also compare against a TF-IDF linear classifier to check whether the recurrent models justify their added cost. These are proposed follow-ups, not measured results in this report.")
    heading('7. Reproducibility and conclusion')
    text('The implementation has five automated tests covering the cell equations and finite gradients, parameter counts, exact-duplicate removal, padding behavior, and equivalence of the vectorized input transform to step-by-step computation. The experiment saves its vocabulary, review hashes, environment, per-epoch results and checkpoints. These files allow the same split and model settings to be inspected without retraining.')
    text('We completed the required three-cell comparison within the available training time. The main practical finding is that loss weighting improves validation macro F1 in this setting, while extra cell complexity does not guarantee a better test result. The models are useful for this classification exercise, but their neutral-class errors remain an important weakness.')
    heading('References')
    for ref in [
        '[1] Course handout. Lab Project on Recurrent Neural Networks, Advanced Natural Language Processing, 2026/2027, pp. 1-2. Guidelines_RNN_2627.pdf.',
        '[2] UCI Machine Learning Repository. Drug Review Dataset (Drugs.com), dataset 462. Description: https://archive.ics.uci.edu/dataset/462 . Archive used: https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip . The description page returned HTTP 403 when checked; the archive was downloaded from the UCI server.',
        '[3] Greff, K. et al. LSTM: A Search Space Odyssey. IEEE Transactions on Neural Networks and Learning Systems, 2017. https://arxiv.org/abs/1503.04069 . CIFG reading listed in the handout.',
        '[4] WMC reading listed in the course appendix: https://www.sciencedirect.com/science/article/pii/S0893608021003439 . Cell equations in this project follow the supplied appendix.',
    ]:
        text(ref, 'small')
    def footer(canvas, doc):
        canvas.setFont('Times-Roman', 8)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(52, 27, 'ANLP 2026/2027 | Drug review classification')
        canvas.drawRightString(543, 27, str(doc.page))
    SimpleDocTemplate(str(pdf_dir/'report.pdf'), pagesize=(595,842), leftMargin=52,
                      rightMargin=52, topMargin=40, bottomMargin=42,
                      title='Drug Review Classification with Vanilla, CIFG and WMC LSTMs',
                      author=members).build(story, onFirstPage=footer, onLaterPages=footer)
    (pdf_dir/'report.md').write_text('\n'.join(markdown), encoding='utf-8')
    print('Built final report:', pdf_dir/'report.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='results/gpu_45min')
    parser.add_argument('--output', default='output')
    args = parser.parse_args()
    build(args.run, args.output)


if __name__ == '__main__':
    main()

"""Non-recurrent reference points evaluated on exactly the same splits as the LSTMs."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .data import LABELS, tokenize


def scores(truth, predictions):
    return {'accuracy': accuracy_score(truth, predictions),
            'macro_f1': f1_score(truth, predictions, labels=[0, 1, 2], average='macro', zero_division=0),
            'report': classification_report(truth, predictions, labels=[0, 1, 2], target_names=LABELS,
                                            output_dict=True, zero_division=0),
            'confusion': confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist()}


def majority(splits):
    label = int(np.argmax(np.bincount([r['label'] for r in splits['train']], minlength=3)))
    truth = [r['label'] for r in splits['test']]
    return {'label': LABELS[label], 'test': scores(truth, [label] * len(truth))}


def tfidf_logreg(splits, seed, grid=(0.25, 1.0, 4.0)):
    """Word 1-2 gram TF-IDF + balanced logistic regression; C is chosen on validation."""
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    x = {k: (vectorizer.fit_transform if k == 'train' else vectorizer.transform)([r['text'] for r in rows])
         for k, rows in [('train', splits['train']), ('val', splits['val']), ('test', splits['test'])]}
    y = {k: [r['label'] for r in splits[k]] for k in x}
    search = []
    for c in grid:
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=seed)
        model.fit(x['train'], y['train'])
        search.append((f1_score(y['val'], model.predict(x['val']), average='macro'), c, model))
    val_f1, c, model = max(search, key=lambda s: s[0])
    return {'C': c, 'val_macro_f1': val_f1, 'features': len(vectorizer.vocabulary_),
            'grid': {str(g): f for f, g, _ in search}, 'test': scores(y['test'], model.predict(x['test']).tolist())}

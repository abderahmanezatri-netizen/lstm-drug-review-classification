import argparse
import json
import torch
from .models import Classifier
from .data import tokenize, LABELS

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--vocab', default=None, help='Defaults to vocab.json beside the checkpoint directory')
    p.add_argument('--text', required=True)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    from pathlib import Path
    vocab_path = Path(args.vocab) if args.vocab else Path(args.checkpoint).parent.parent / 'vocab.json'
    vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
    model = Classifier(checkpoint['vocab_size'], checkpoint['variant'], checkpoint['embedding_dim'],
                       checkpoint['hidden_size'], pooling=checkpoint.get('pooling', 'last'))
    if checkpoint.get('vocab_sha256'):
        import hashlib
        if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != checkpoint['vocab_sha256']:
            raise ValueError('This vocabulary does not belong to this checkpoint.')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    ids = [vocab.get(t, 1) for t in tokenize(args.text)[:checkpoint['max_length']]] or [1]
    with torch.no_grad():
        probabilities = model(torch.tensor([ids]), torch.tensor([len(ids)])).softmax(-1)[0]
    print(json.dumps(dict(zip(LABELS, probabilities.tolist())), indent=2))

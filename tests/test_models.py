import unittest
import torch
from src.models import LSTMCell, Classifier


class ModelTests(unittest.TestCase):
    def test_equations(self):
        torch.manual_seed(3)
        x, h, c = torch.randn(2, 4), torch.randn(2, 3), torch.randn(2, 3)
        for variant in ['vanilla', 'cifg', 'wmc']:
            cell = LSTMCell(4, 3, variant)
            values = (cell.x(x) + cell.h(h)).chunk(3 if variant == 'cifg' else 4, -1)
            if variant == 'cifg':
                f, o, g = values
                f = f.sigmoid()
                i = 1 - f
            else:
                i, f, o, g = values
                i = (i + cell.ci(c).tanh() if variant == 'wmc' else i).sigmoid()
                f = (f + cell.cf(c).tanh() if variant == 'wmc' else f).sigmoid()
            expected_c = f*c + i*g.tanh()
            if variant == 'wmc':
                o = o + cell.co(expected_c).tanh()
            expected_h = o.sigmoid()*expected_c.tanh()
            actual_h, actual_c = cell(x, (h, c))
            torch.testing.assert_close(actual_h, expected_h)
            torch.testing.assert_close(actual_c, expected_c)
            (actual_h.sum() + actual_c.sum()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters()))

    def test_padding_and_parameters(self):
        for variant in ['vanilla', 'cifg', 'wmc']:
            model = Classifier(20, variant, 4, 3).eval()
            torch.testing.assert_close(model(torch.tensor([[2,3]]), torch.tensor([2])),
                                       model(torch.tensor([[2,3,0,0]]), torch.tensor([2])))
            gates = 3 if variant == 'cifg' else 4
            expected = 20*4 + gates*3*(4+3+1) + 3*3+3 + (3*3*3 if variant == 'wmc' else 0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_vectorized_input_matches_step_by_step_outputs_and_gradients(self):
        import copy
        torch.manual_seed(71)
        tokens = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
        lengths = torch.tensor([3, 4])
        for variant in ['vanilla', 'cifg', 'wmc']:
            for pooling in ['last', 'mean']:
                optimized = Classifier(20, variant, 4, 3, dropout=0., pooling=pooling)
                reference = copy.deepcopy(optimized)
                embeddings = reference.embedding(tokens)
                h = embeddings.new_zeros(2, 3)
                c, total = torch.zeros_like(h), torch.zeros_like(h)
                for t in range(tokens.size(1)):
                    new_h, new_c = reference.cell(embeddings[:, t], (h, c))
                    active = (t < lengths).unsqueeze(1)
                    h, c = torch.where(active, new_h, h), torch.where(active, new_c, c)
                    total = total + h * active
                if pooling == 'mean':
                    h = total / lengths.unsqueeze(1)
                expected = reference.output(h)
                actual = optimized(tokens, lengths)
                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                expected.sum().backward()
                for left, right in zip(optimized.parameters(), reference.parameters()):
                    torch.testing.assert_close(left.grad, right.grad)

    def test_mean_pooling_ignores_padding(self):
        for variant in ['vanilla', 'cifg', 'wmc']:
            model = Classifier(20, variant, 4, 3, pooling='mean').eval()
            torch.testing.assert_close(model(torch.tensor([[2, 3]]), torch.tensor([2])),
                                       model(torch.tensor([[2, 3, 0, 0]]), torch.tensor([2])))


if __name__ == '__main__':
    unittest.main()

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from src.data import load_raw, tokenize


class DataTests(unittest.TestCase):
    def test_deduplication_across_official_boundary(self):
        temporary_root = Path(__file__).resolve().parents[1] / 'tmp' / 'tests'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            def tsv(rows):
                stream = io.StringIO()
                writer = csv.writer(stream, delimiter='\t')
                writer.writerow(['', 'review', 'rating'])
                writer.writerows(rows)
                return stream.getvalue()
            with zipfile.ZipFile(Path(directory)/'drugs.zip','w') as z:
                z.writestr('drugsComTrain_raw.tsv',tsv([[1,'GOOD &amp; helpful',9],[2,'good & helpful',8],[3,'bad',1]]))
                z.writestr('drugsComTest_raw.tsv',tsv([[4,'GOOD & helpful',10],[5,'Mixed results',5],[6,'Excellent',10]]))
            raw = load_raw(directory)
            self.assertEqual(raw['train_removed'],1)
            self.assertEqual(raw['test_removed'],1)
            self.assertEqual([r['label'] for r in raw['test']],[1,2])
            self.assertFalse({r['key'] for r in raw['train']} & {r['key'] for r in raw['test']})
            self.assertEqual(tokenize("It didn&#39;t help!"),['it',"didn't",'help','!'])


if __name__ == '__main__':
    unittest.main()

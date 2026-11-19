import importlib.util, json, sys, shutil, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/improvement/prepare_raw_inputs.py'
spec = importlib.util.spec_from_file_location('stager', path)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class StagerTests(unittest.TestCase):
    def setup_fixture(self, directory):
        base = Path(directory)
        self.source = base/'raw'
        self.dest = base/'snapshot'
        self.manifest = {f'{role}_events': [f'{i}/20200101.txt'] for i, role in enumerate(p.COUNTS)}
        for i in range(3):
            folder = self.source/'event'/str(i)/'20200101.txt'
            folder.mkdir(parents=True)
            (folder/'20200101.txt_full_summary.txt').write_text('test fixture only')
        price = self.source/'series'/'SP500.csv'
        price.parent.mkdir()
        price.write_text('prices fixture only')
        self.compute = p.CANONICAL_CONTENT_SHA256

        def freeze(manifest, root, price, target):
            target.mkdir()
            shutil.copytree(root, target/'event')
            shutil.copytree(price.parent, target/'series')
            shutil.copy2(manifest, target/'manifest.json')
            (target/'snapshot.json').write_text('{}')
            for d in target.rglob('*'):
                if d.is_dir():
                    d.chmod(0o500)
            target.chmod(0o500)

        def verify(target):
            assert (target/'snapshot.json').exists()
            return dict(manifest_sha256=p.MANIFEST_SHA256, content_sha256=self.compute,
                        event_folder_count=3, price_sha256=p.EXPECTED_PRICE_SHA256)

        def build(*args, **kwargs):
            self.assertEqual(kwargs['timing_policy'], p.TIMING_POLICY)
            self.assertFalse(kwargs['skip_missing'])
            d = {f'{r}_set': types.SimpleNamespace(event_ids=self.manifest[f'{r}_events'])
                 for r in p.COUNTS}
            return types.SimpleNamespace(**d, audit=dict(emitted_counts=p.COUNTS,
                    excluded_counts={}, covered_bars={r: 175 for r in p.COUNTS}))

        self.freeze = freeze
        self.patches = (patch.object(p, 'COUNTS', dict(train=1, vali=1, test=1)),
                        patch.object(p, 'canonical_manifest_bytes',
                                     lambda _: json.dumps(self.manifest).encode()),
                        patch.dict(sys.modules, {'data.c1_protocol': types.SimpleNamespace(
                            build_c1_dataloaders=build, event_key=lambda s: (None,None,s)),
                            'scripts.freeze_c1_inputs': types.SimpleNamespace(freeze=freeze, verify=verify)}))
        for change in self.patches:
            change.start()
            self.addCleanup(change.stop)

    def test_canonical_then_no_clobber(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.setup_fixture(workspace)
            actual = p.prepare(self.source, self.dest, False)
            self.assertEqual(actual['classification'], 'canonical')
            sidecar = self.dest.with_name('snapshot.receipt.json')
            self.assertEqual(json.loads(sidecar.read_text()), actual)
            with self.assertRaisesRegex(FileExistsError, 'refusing to replace'):
                p.prepare(self.source, self.dest, False)
            p._unlock_staging_for_cleanup(self.dest)

    def test_alternate_rejected_unless_exploratory(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.setup_fixture(workspace)
            self.compute = 'e'*64
            with self.assertRaisesRegex(ValueError, 'No snapshot was published'):
                p.prepare(self.source, self.dest, False)
            self.assertEqual({q.name for q in self.dest.parent.iterdir()}, {'raw'})
            self.assertEqual(p.prepare(self.source, self.dest, True)['classification'], 'exploratory')
            p._unlock_staging_for_cleanup(self.dest)

    def test_missing_summary_rejected_even_exploratory(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.setup_fixture(workspace)
            (self.source/'event'/'1'/'20200101.txt'/'20200101.txt_full_summary.txt').write_text('  ')
            with self.assertRaisesRegex(ValueError, 'empty summary'):
                p.prepare(self.source, self.dest, True)
            self.assertFalse(self.dest.exists())

    def test_destination_within_source_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.setup_fixture(workspace)
            with self.assertRaisesRegex(ValueError,'outside the source'):
                p.prepare(self.source, self.source/'nested', True)

    def test_git_blob_hash_before_worktree(self):
        with patch.object(p, 'MANIFEST_SHA256', p.sha256(b'example\n')), patch.object(
                p.subprocess, 'run', return_value=types.SimpleNamespace(stdout=b'example\r\n')):
            self.assertEqual(self._call_actual_manifest(b'example\r\n'), b'example\n')
            with self.assertRaises(ValueError):
                self._call_actual_manifest(b'not correct')

    def _call_actual_manifest(self, payload):
        with patch.object(p.subprocess, 'run', return_value=types.SimpleNamespace(stdout=payload)):
            return p.canonical_manifest_bytes(path.parent)

if __name__ == '__main__':
    unittest.main()

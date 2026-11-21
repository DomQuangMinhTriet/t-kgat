"""Verify the uploaded source snapshot and current recovery without training."""

import ast
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path.cwd()
results = root / 'tkgat-lastfm-results'
target = root / 'data/export/kaggle/tkgat-lastfm-kgat-recovery.zip'
with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
    for path in sorted(results.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(results).as_posix())
with ZipFile(target) as archive:
    assert archive.testzip() is None
    assert 'results/runs/lastfm-u25-latest/bprmf_summary.json' in archive.namelist()

notebook = json.loads((root / 'notebooks/05_kaggle_kgat_resume.ipynb').read_text(encoding='utf-8'))
namespace = {}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    tree = ast.parse(source)
    if 'def recovery_archive(' in source:
        nodes = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'resume-helpers', 'exec'), namespace)

with tempfile.TemporaryDirectory(prefix='verify-kgat-', dir=root / 'tmp') as temporary:
    workspace = Path(temporary)
    payload = workspace / 'payload'
    with ZipFile(root / 'data/export/kaggle/tkgat-lastfm-latest-kaggle.zip') as archive:
        archive.extractall(payload)
    project = workspace / 'project'
    subprocess.run(['git', 'clone', '-q', str(payload / 'source.bundle'), str(project)], check=True)
    spec = importlib.util.spec_from_file_location('frozen_kaggle_runtime', project / 'scripts/kaggle_lastfm.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.prepare(payload)
    identity = runtime.identity()
    selected = namespace['recovery_archive'](workspace, workspace, identity, target)
    runtime.extract_results(selected)
    namespace['require_baselines']({'complete': runtime.complete})
    assert not runtime.complete('kgat')
    for name in ('registry.csv', 'kaggle-run.json'):
        assert (project / 'results' / name).read_bytes() == (results / 'results' / name).read_bytes()
    print('Verified unchanged registry and source snapshot:', identity['source_commit'])
    print('Next model: KGAT; no model was trained.')
print('Recovery upload:', target, 'bytes:', target.stat().st_size)

"""Check packaged bytes and available historical receipts without importing experiment code."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    failures, missing, receipt_checks = [], [], []
    files = []
    for line in (ROOT/'ARTIFACTS.sha256').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        p = ROOT/relative
        if not p.is_file() or sha(p) != expected:
            failures.append('Packaged hash mismatch: '+relative)
        files.append(p)
    with (ROOT/'SOURCE_MAP.csv').open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if sha(ROOT/r['repository_path']) != r['repository_sha256']:
                failures.append('Source map mismatch: '+r['repository_path'])
    for p in files:
        if not p.is_file():
            continue
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        elif p.suffix in ('.json', '.ipynb'):
            value = json.loads(p.read_text(encoding='utf-8-sig'))
            if p.suffix == '.ipynb':
                for c in value['cells']:
                    if c['cell_type'] == 'code' and (c.get('outputs') or c.get('execution_count') is not None):
                        failures.append('Notebook contains saved execution output: '+p.relative_to(ROOT).as_posix())

    def verify(path, expected, origin):
        if not path.is_file():
            missing.append(dict(path=path.relative_to(ROOT).as_posix(), receipt=origin))
        elif sha(path) != expected:
            failures.append('Execution-receipt mismatch: '+path.relative_to(ROOT).as_posix())
        else:
            receipt_checks.append(path.relative_to(ROOT).as_posix())

    for folder, receipt, key in [
        ('evidence/development/models','FIT_COMPLETE.json','files'),
        ('evidence/tbx11k','EVALUATION_COMPLETE.json','results_sha256'),
        ('evidence/comparators','ANALYSIS_COMPLETE.json','files'),
        ('evidence/matched_resolution','COMPLETE.json','files'),
        ('evidence/preprocessing','COMPLETE.json','files')]:
        value = json.loads((ROOT/folder/receipt).read_text())
        for name, expected in value[key].items():
            target = ROOT/folder/name
            if folder == 'evidence/preprocessing' and name == 'audit_dino_preprocessing.py':
                target = ROOT/'experiments/preprocessing'/name
            verify(target, expected, folder+'/'+receipt)
    provenance = json.loads((ROOT/'evidence/development/results/evaluation_provenance.json').read_text())
    for name in ['prepare','experiment']:
        verify(ROOT/f'experiments/development/{name}.py', provenance['environment'][name+'_code_sha256'], 'development evaluation provenance')
    verify(ROOT/'evidence/development/study/manifest.csv', provenance['manifest_sha256'], 'development evaluation provenance')
    for code, receipt in [
        ('experiments/comparators/compare_saved_models.py','evidence/comparators/ANALYSIS_COMPLETE.json'),
        ('experiments/matched_resolution/analyse_matched_resolution.py','evidence/matched_resolution/COMPLETE.json')]:
        verify(ROOT/code, json.loads((ROOT/receipt).read_text())['code_sha256'], receipt)
    tbx = json.loads((ROOT/'evidence/tbx11k/EVALUATION_COMPLETE.json').read_text())
    verify(ROOT/'evidence/tbx11k_audit/eligible_manifest.csv', tbx['manifest_sha256'], 'TBX11K evaluation completion')
    verify(ROOT/'evidence/tbx11k/protocol', tbx['protocol_sha256'], 'TBX11K evaluation completion')
    report = dict(status='failed' if failures else 'passed', packaged_files=len(files),
                  available_receipt_checks=len(receipt_checks), missing_receipt_assets=missing,
                  failures=failures,
                  scope='Checks packaged files and available receipt assets. Missing archive assets are reported; completeness is not asserted.')
    out = ROOT/'verification-output'
    out.mkdir(exist_ok=True)
    (out/'artifact_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:(len(v) if k=='missing_receipt_assets' else v) for k,v in report.items()}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

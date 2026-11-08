"""Regenerate every figure, table and analysis output of the article from saved scores.

Usage (from the repository root):  python analysis/reproduce.py [OUT]

Runs each script in analysis/ in order, writes the outputs to OUT
(default: analysis-output/), and checks that the regenerated analysis
files and table bodies are byte-identical to those saved in
evidence/analysis/. Needs Python 3.10+, NumPy and Matplotlib; no GPU,
data download or model fitting.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / 'analysis'
SAVED = ROOT / 'evidence' / 'analysis'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'analysis-output'
OUT.mkdir(parents=True, exist_ok=True)

def run(script, *args):
    print(f'> {script}', flush=True)
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)

U, C, S = OUT / 'threshold_uncertainty.json', OUT / 'threshold_calibration.json', OUT / 'local_threshold_subgroups.json'
run('threshold_uncertainty.py', ROOT, OUT)          # Sections 3.4, 3.6; results 4.2, 4.4-4.5
run('threshold_calibration.py', ROOT, OUT)          # Section 3.5 (Algorithm 1); results 4.3
run('figure1_study_design.py', OUT)                 # Figure 1
run('figure2_external_performance.py', ROOT, OUT)   # Figure 2
run('figures_3_5_S1.py', ROOT, U, OUT)              # Figures 3, 5 and S1
run('figure4_local_calibration.py', C, OUT)         # Figure 4
run('tables.py', ROOT, U, S, C, OUT)                # Tables 2-4, S4-S5

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks = [(OUT / n, SAVED / n) for n in ['threshold_uncertainty.json', 'threshold_calibration.json', 'local_threshold_subgroups.json']]
checks += [(p, SAVED / 'tables' / p.name) for p in sorted((OUT / 'tables').glob('*.tex'))]
failed = [new.name for new, old in checks if sha(new) != sha(old)]

# Plotted estimates: each recorded source file must be present, unchanged, in evidence/.
prov = json.loads((SAVED / 'estimates_provenance.json').read_text(encoding='utf-8'))
have = {sha(p) for p in (ROOT / 'evidence').rglob('*') if p.is_file()}
missing = [k for k, v in prov['sources'].items() if v['sha256'] not in have]

print(f'\n{len(checks) - len(failed)}/{len(checks)} outputs identical to evidence/analysis/')
print(f'{len(prov["sources"]) - len(missing)}/{len(prov["sources"])} plotted-estimate source files found unchanged in evidence/')
print(f'Figures written to {OUT}: ' + ', '.join(sorted(p.name for p in OUT.glob('Fig*.pdf'))))
if failed or missing:
    print('MISMATCH:', failed + missing); sys.exit(1)

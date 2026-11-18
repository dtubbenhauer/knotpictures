"""Recompute the paper's analyses in a new destination directory from published inputs and generated drawings."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path,help='New directory outside the original package')
    parser.add_argument('--retrain',action='store_true',help='Refit the six CNNs; otherwise reuse the supplied checkpoints')
    args=parser.parse_args()
    dest=args.destination.resolve()
    if dest==ROOT or ROOT in dest.parents or dest.exists():
        parser.error('Choose a new directory outside the original package.')
    from fetch_inputs import SPECS, ready
    missing = [name for name in SPECS if not ready(name)]
    if missing:
        parser.error('Run python code/fetch_inputs.py first. Missing or unverified: '+', '.join(missing))
    shutil.copytree(ROOT,dest)
    def run(folder,script,*arguments):
        print(f'[{folder}] {script} '+ ' '.join(arguments),flush=True)
        subprocess.run([sys.executable,'-B',script,*arguments],cwd=dest/folder,env=ENV,check=True)
    tar='knot_crossings_dataset.tar.gz'
    run('corpus','table_baseline.py')
    run('legacy','legacy_audit.py','--dataset',tar,'--paper','paper_illustrations.zip','--out','results')
    run('legacy','overlap_audit.py','--manifest','results/released_counts.csv','--out','results')
    run('legacy','poster_census.py','--poster','knottable_v1.svg','--out','results')
    run('legacy','poster_figure.py','--poster','knottable_v1.svg','--results','results','--out','results')
    run('legacy','legacy_figure.py','--paper','paper_illustrations.zip','--out','results')
    run('paired','generate_pairs.py')
    run('paired','summarize.py')
    run('paired','measure_faces.py')
    run('paired','measure_mixed.py')
    run('paired','measure_poster.py')
    run('paired','measure_jkasd.py')
    run('visuals','figures.py')
    run('visuals','mixed_junctions.py')
    run('visuals','camera-views.py')
    run('visuals','invariant_states.py')
    run('visuals','trefoil-mini.py')
    run('visuals','draw_culprit.py')
    run('visuals','reading-figures.py','--archive','../legacy/paper_illustrations.zip')
    run('visuals','poster-transfer.py')
    run('replay','ink_features.py','../legacy/'+tar,'../corpus/manifest.csv','../corpus/knots.png','.')
    run('replay','source_folds.py','.')
    run('replay','stability.py','.')
    run('replay','validate_lineage.py','../legacy/'+tar,'.')
    run('replay','validate_source_blocks.py','../legacy/'+tar,'.')
    if args.retrain:
        run('learning','archive.py','../legacy/'+tar)
        run('learning','cnn.py','../paired/manifest.csv','--out','results')
    run('learning','matched.py','../legacy/'+tar)
    run('learning','cross_paired.py','../paired/manifest.csv')
    run('learning','transfer.py','../corpus/manifest.csv','--models','archive_results',
        '--out','archive_results','--training-set','archive')
    run('learning','transfer.py','../corpus/manifest.csv','--models','results','--out','results')
    run('replay','matched_ink_control.py','.','../learning/archive_results')
    run('.','verify.py','--measurements-only')
    print(f'Finished. Regenerated results: {dest}')

if __name__=='__main__':
    main()

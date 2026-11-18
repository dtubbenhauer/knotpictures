"""Verify checksums and recompute manuscript scores from individual observations.

Uses Python's standard library. Pass --raw to check all PNGs inside the archived
training/test tarball against the exact files used to compute the measurements.
This command never modifies the package.
"""
import argparse
import csv
import hashlib
import json
from zipfile import ZipFile
import tarfile
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
SIZES = (64, 128, 256, 512, 1024, 2048)
SEEDS = (17, 29, 43)

def rows(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def require(condition, msg):
    if not condition:
        raise RuntimeError(msg)

def digest(f):
    return hashlib.file_digest(f, 'sha256').hexdigest()

def by_int(table, key, value):
    return [r for r in table if int(r[key]) == value]

def hits(table, target, pred):
    return sum(int(r[target]) == int(r[pred]) for r in table)

def show(name, actual, expected):
    require(actual == expected, f'{name}: expected {expected}, got {actual}')
    print(f'{name}: {actual}')

def check_integrity(raw):
    manifest = json.loads((ROOT.parent/'package-manifest.json').read_text(encoding='utf-8'))
    for rec in manifest['files']:
        path = ROOT.parent/rec['path']
        with path.open('rb') as f:
            require(digest(f) == rec['sha256'], f'Checksum mismatch for {path}')
    print(f"Verified {len(manifest['files'])} distributed file checksums.")
    if raw:
        with (ROOT/'legacy/knot_crossings_dataset.tar.gz').open('rb') as pkg:
            expected = json.loads((ROOT/'inputs.json').read_text())['dataset']['sha256']
            require(digest(pkg)==expected, 'Raw tarball checksum mismatch.')
        released = {r['file']: r['sha256'] for r in rows('legacy/results/released_counts.csv')}
        found = set()
        with tarfile.open(ROOT/'legacy/knot_crossings_dataset.tar.gz', 'r|gz') as archive:
            for member in archive:
                if member.name in released:
                    with archive.extractfile(member) as f:
                        require(digest(f) == released[member.name], f'Archive mismatch: {member.name}')
                    found.add(member.name)
        show('Archived PNG bytes', len(found), 9100)

def check_rasters():
    chart = rows('corpus/observations.csv')
    show('Jkasd table, correct nontrivial drawings at 230,160,128,96,64,48,32',
         [hits([r for r in by_int(chart, 'side_px', size) if int(r['expected']) > 0], 'expected','prediction')
          for size in (230,160,128,96,64,48,32)], [14,14,14,14,11,5,0])
    require(all(r['prediction']=='0' for r in chart if r['expected']=='0'), 'Circle error')
    phase = rows('corpus/phase-observations.csv')
    for size,expected in [(64,(11,12)),(48,(4,5))]:
        values=[hits([r for r in by_int(phase,'side_px',size) if int(r['expected']) and
                      r['dx']==str(dx) and r['dy']==str(dy)],'expected','prediction')
                for dx in (0,1) for dy in (0,1)]
        show(f'Jkasd table shifted crops at {size}',(min(values),max(values)),expected)
    poster=rows('legacy/results/poster_counts.csv')
    show('Rideout poster, arc counter at 64..2048',
         [hits(by_int(poster,'width',s),'label','prediction_otsu') for s in SIZES],
         [1,5,35,62,82,88])
    euler=rows('paired/poster-euler.csv')
    show('Rideout poster, full Euler estimate at 64..2048',
         [hits(by_int(euler,'width',s),'count','euler_prediction') for s in SIZES],
         [23,75,77,81,73,55])
    independent=rows('paired/jkasd-euler.csv')
    show('Jkasd Euler, correct at 32,48,64,96,128,160,230',
         [hits(by_int(independent,'width',s),'count','euler_prediction') for s in (32,48,64,96,128,160,230)],
         [11,15,15,15,15,15,15])
    misses=[r for r in by_int(euler,'width',2048) if r['count']!=r['euler_prediction']]
    show('Extra tips on the 33 high-resolution misses',
         Counter(int(r['tips'])-2*int(r['count']) for r in misses),Counter({1:31,2:2}))
    release=rows('legacy/results/released_counts.csv')
    train=[r for r in release if r['file'].startswith('train/')]
    hashes={r['sha256'] for r in train}
    test=[r for r in release if r['file'].startswith('test/')]
    show('Archive: fitting pool, distinct PNG bytes, test pool, overlapping test files',
         (len(train),len(hashes),len(test),sum(r['sha256'] in hashes for r in test)),
         (9051,6657,49,18))
    show('Archive: correct arc counts on 9051 resized/transformed files',
         hits(train,'label','prediction_otsu_1'),65)
    names={Path(r['file']).stem:int(r['components_otsu_1']) for r in rows('legacy/results/figure_counts_release.csv')}
    show('Illustrated nine-crossing image, original and final', (names['knot_orig'],names['knot_final']), (9,161))
    lineage=rows('replay/resize_lineage_checks.csv')
    show('Scale descendants, maximum deviation in any 8-bit pixel',
         (len(lineage),max(int(r['max_8bit_error']) for r in lineage)),(8620,1))
    blocks=rows('replay/source_block_checks.csv')
    show('Within-source radial profile checks', (sum(int(r['exact']) for r in blocks),len(blocks)),(199,199))
    tiling=rows('visuals/mixed-junctions.csv')
    show('Trefoil crossing count at four gap settings',
         [int(r['endpoints'])-int(r['components'])+int(r['bounded_white_regions']) for r in tiling],
         [3]*4)

def check_ink():
    allpred=rows('replay/global_ink_predictions.csv')
    random=[r for r in allpred if r['evaluation']=='image_split']
    eligible=[r for r in random if int(r['expected'])>=5]
    source=rows('replay/source_fold_predictions.csv')
    for name,subset,score in [('Random images',random,(1810,1811)),('Random images, labels 5..10',eligible,(1192,1193)),('Leave one original drawing out',source,(1292,5964))]:
        show(name,(hits(subset,'expected','prediction'),len(subset)),score)
    source_ids={r['source_id'] for r in source}
    per_source=[[r for r in source if r['source_id']==sid] for sid in source_ids]
    show('Source-weighted accuracy, percent',round(100*mean(hits(s,'expected','prediction')/len(s) for s in per_source),2),36.42)
    f=rows('replay/global_ink_features.csv')
    show('Median fractional ink percent, archive and external table',
         [round(100*median(float(r['fractional_ink']) for r in f if r['source']==where),2) for where in ['train','printed_table']],[0.43,7.01])
    show('Distinct omitted originals', len({r['source_id'] for r in source}),85)
    show('Exact copies in seed 42 split',sum(int(r['same_file_as_training']) for r in random),621)
    stability=rows('replay/stability_predictions.csv')
    for split,expected in [('image',(18110,18110)),('sha256',(18072,18075))]:
        subset=[r for r in stability if r['evaluation'].startswith(split+'_')]
        show(f'Ten {split} splits, pooled correct and total',(hits(subset,'expected','prediction'),len(subset)),expected)
    external=[r for r in allpred if r['evaluation']=='table']
    show('Ink control on external table, correct and total', (hits(external,'expected','prediction'),len(external)), (0,15))

def check_matched():
    expected={'moments':[(1011,0.04),(169,1.94)],17:[(552,.63),(198,1.42)],29:[(341,.94),(118,1.79)],43:[(396,.77),(166,1.36)]}
    matched=rows('learning/archive_results/matched_17.csv')
    seen=[r for r in matched if r['arm']=='seen_source']
    unseen=[r for r in matched if r['arm']=='unseen_source']
    assignment=rows('learning/archive_results/archive_split.csv')
    fit_hashes={r['sha256'] for r in assignment if r['split'] in ('train','validation','val')}
    require(Counter((r['count'],r['scale']) for r in seen)==Counter((r['count'],r['scale']) for r in unseen),'Label/scale mismatch')
    show('Matched distinct evaluation hashes',len({r['sha256'] for r in matched}),2108)
    require(all(r['sha256'] not in fit_hashes for r in matched),'Training hash leaked')
    for method in ['moments',*SEEDS]:
        if method=='moments':
            data=rows('replay/matched_ink_predictions.csv')
            arm,pred,target='evaluation','prediction','expected'
        else:
            data=rows(f'learning/archive_results/matched_{method}.csv')
            arm,pred,target='arm','rounded','count'
        values=[]
        for a in ('seen_source','unseen_source'):
            sub=[r for r in data if r[arm]==a]
            require(len(sub)==1054,f'{a} has {len(sub)} rows')
            values.append((hits(sub,target,pred),round(mean(abs(int(r[pred])-int(r[target])) for r in sub),2)))
        show(f'{method} familiar/unseen (correct, MAE)',values,expected[method])

def check_pairs():
    main=rows('paired/manifest.csv')
    require(all(r['count']==r['geometric_count'] for r in main),'Polygon intersection labels disagree')
    show('Clean/curl underlying geometries',len({(r['group'],r['k'],r['curl']) for r in main}),540)
    mixedfiles=rows('paired/mixed-images.csv')
    expected_images = {r['image'] for r in mixedfiles} | {r['image'] for r in main}
    paired = ROOT/'paired'
    if (paired/'images').is_dir():
        found_images = {p.relative_to(paired).as_posix() for p in (paired/'images').rglob('*.png')}
    else:
        with ZipFile(paired/'drawings.zip') as archived:
            require(archived.testzip() is None, 'Synthetic PNG archive CRC mismatch')
            found_images = set(archived.namelist())
    require(found_images == expected_images, 'Generated PNG manifest disagrees with supplied drawings')
    show('Individual generated PNGs', len(found_images), 1020)
    show('Individual mixed-gap PNGs',sum(r['image'] in found_images for r in mixedfiles),160)
    mixed=rows('paired/mixed-counts.csv')
    expected={'clean':(150,5,158,80),'break':(13,0,0,80),'thin':(16,125,125,77),'mixed':(23,17,146,80)}
    for style,answer in expected.items():
        subset=[r for r in mixed if r['style']==style]
        pair={(r['group'],r['k'],r['curl']):int(r['estimate']) for r in subset}
        require(len(subset)==160,'Incorrect test size')
        increment=sum(pair[(g,k,'1')]-value==1 for (g,k,curl),value in pair.items() if curl=='0')
        faces=sum(int(r['white_bounded'])-1==int(r['count']) for r in subset)
        formula=[int(r['tips'])+int(r['white_bounded'])-int(r['ink_components']) for r in subset]
        require(all(int(r['estimate'])==p for r,p in zip(subset,formula)),'Euler readout inconsistent')
        show(f'{style}: arc hits, face hits, Euler hits, Euler +1 pairs',
             (hits(subset,'count','ink_components'),faces,hits(subset,'count','estimate'),increment),answer)
    interior=[r for r in mixed if r['style']=='mixed' and 0<int(r['opened'])<int(r['count'])]
    show('Genuinely mixed cases (correct, total)',(hits(interior,'count','estimate'),len(interior)),(117,122))
    for folder in ('results','archive_results'):
        saved=list((ROOT/'learning'/folder).glob('model_*.npz'))
        show(f'CNN checkpoints in {folder}',sorted(p.stem for p in saved),[f'model_{s}' for s in SEEDS])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',action='store_true',help='Verify the hash of each archived PNG member')
    parser.add_argument('--measurements-only', action='store_true', help='Check recalculated tables after regeneration; generated PDF metadata may vary')
    options=parser.parse_args()
    if not options.measurements_only:
        check_integrity(options.raw)
    check_rasters()
    check_ink()
    check_matched()
    check_pairs()
    print('All checks passed.')

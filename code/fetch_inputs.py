"""Fetch and checksum published inputs for the knot image experiments. Python 3.11+."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent
SPECS = json.loads((ROOT / 'inputs.json').read_text(encoding='utf-8'))
PAPER_ZIP = ROOT / 'legacy' / 'paper_illustrations.zip'
KNOT_TABLE = ROOT / 'corpus' / 'knots.png'


def matches(path, sha):
    if not path.is_file():
        return False
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest() == sha


def ready(kind):
    spec = SPECS[kind]
    if kind != 'paper':
        return matches(ROOT/spec['output'],spec['sha256'])
    hashes = spec['members']
    if not matches(KNOT_TABLE, hashes['figs/knots.png']) or not PAPER_ZIP.is_file():
        return False
    try:
        with ZipFile(PAPER_ZIP) as archive:
            for member, expected in hashes.items():
                if member != 'figs/knots.png':
                    with archive.open('arXiv/'+member) as data:
                        if hashlib.file_digest(data,'sha256').hexdigest() != expected:
                            return False
    except (BadZipFile, KeyError, OSError):
        return False
    return True


def unpack_paper(path):
    hashes = SPECS['paper']['members']
    data = {}
    with tarfile.open(path, 'r:gz') as archive:
        for name, expected in hashes.items():
            with archive.extractfile(name) as stream:
                content = stream.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Unexpected source illustration bytes: {name}')
            data[name] = content
    with NamedTemporaryFile(dir=KNOT_TABLE.parent, delete=False) as temp:
        src_table = Path(temp.name)
        temp.write(data.pop('figs/knots.png'))
    try:
        with NamedTemporaryFile(dir=PAPER_ZIP.parent, delete=False) as temp:
            src_zip = Path(temp.name)
        try:
            with ZipFile(src_zip, 'w') as archive:
                for name, content in sorted(data.items()):
                    info = ZipInfo('arXiv/'+name, date_time=(1980,1,1,0,0,0))
                    archive.writestr(info, content, compress_type=ZIP_DEFLATED)
            os.replace(src_zip, PAPER_ZIP)
            os.replace(src_table, KNOT_TABLE)
        finally:
            src_zip.unlink(missing_ok=True)
    finally:
        src_table.unlink(missing_ok=True)


def fetch(kind):
    spec = SPECS[kind]
    if ready(kind):
        print(f'{kind}: locally verified')
        return
    # Every destination is a generated input cache; downloaded bytes are checked before install.
    with NamedTemporaryFile(dir=ROOT, prefix='.download-', delete=False) as tmp:
        path = Path(tmp.name)
    try:
        with path.open('wb') as tmp:
            req = Request(spec['url'], headers={'User-Agent':'knotpictures-reproduction/1.0'})
            with urlopen(req,timeout=90) as response:
                while block := response.read(1024*1024):
                    tmp.write(block)
        if path.stat().st_size != spec['bytes'] or not matches(path,spec['sha256']):
            raise ValueError(f'Downloaded {kind} failed its pinned SHA-256 or size check')
        if kind=='paper':
            unpack_paper(path)
        else:
            os.replace(path, ROOT/spec['output'])
        if not ready(kind):
            raise ValueError(f'{kind} failed its installed input check')
        print(f'{kind}: fetched and verified')
    finally:
        path.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='*', metavar='INPUT', help='dataset, poster, or paper (default: all)')
    p.add_argument('--check',action='store_true',help='Verify local inputs without downloading')
    args=p.parse_args()
    for kind in args.inputs or SPECS:
        if kind not in SPECS:
            p.error(f'Unknown input {kind!r}; choose dataset, poster, or paper')
        if args.check:
            if not ready(kind): p.exit(1,f'{kind}: absent or different from its published digest\n')
            print(f'{kind}: locally verified')
        else:
            fetch(kind)

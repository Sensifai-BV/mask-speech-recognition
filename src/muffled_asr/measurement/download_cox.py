"""Download the acoustic subset of Cox 2022 with publisher MD5 verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

NAMES = {'measurements_from_august_2021.zip', 'measurements_from_june_2021.zip',
         'measurements_from_march_2021.zip', 'human_tests_on_masks.zip',
         'description of files.docx'}

def verified(path, entry):
    if not path.exists() or path.stat().st_size != entry['size']:
        return False
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'md5').hexdigest()
    return digest == (entry.get('computed_md5') or entry['supplied_md5'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('datasets/original/cox-2022'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://api.figshare.com/v2/articles/19174235', timeout=60) as r:
        meta = json.load(r)
    (args.out/'metadata.json').write_text(json.dumps(meta, indent=2))
    def fetch(entry):
        dest = args.out/entry['name']; part = dest.with_suffix(dest.suffix+'.part')
        if verified(dest, entry):
            return f'{dest.name}: verified existing file'
        if not verified(part, entry):
            with urllib.request.urlopen(entry['download_url'], timeout=120) as r, part.open('wb') as w:
                while block := r.read(1024*1024):
                    w.write(block)
        if not verified(part, entry):
            raise ValueError(f'Checksum or size mismatch: {part}')
        part.replace(dest)
        return f'{dest.name}: downloaded and verified'
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, [f for f in meta['files'] if f['name'] in NAMES]):
            print(result, flush=True)

if __name__ == '__main__':
    main()

"""Build a reproducible 256px image cache from the original DiaMOS Pear.zip."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
EXPECTED = {'healthy':43, 'spot':884, 'curl':54, 'slug':2025}
MD5 = '0f99bc411e2f5e73ac34d1fd99c5ac88'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT/'data'/'Pear.zip')
    args = parser.parse_args()
    digest = hashlib.md5()
    with args.archive.open('rb') as source:
        for chunk in iter(lambda: source.read(16*1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != MD5:
        raise ValueError('Archive checksum mismatch: ' + digest.hexdigest())
    out = ROOT/'data'/'leaves_256'
    manifest = []; counts = {k:0 for k in EXPECTED}
    with ZipFile(args.archive) as z:
        for item in z.infolist():
            path=PurePosixPath(item.filename.replace('\\','/'))
            parts=[part.lower() for part in path.parts]
            if item.is_dir() or path.name.startswith('._') or '__macosx' in parts or 'leaves' not in parts or path.suffix.lower() not in ('.jpg','.jpeg','.png'):
                continue
            after=parts[parts.index('leaves')+1:-1]
            matches=[part for part in after if part in EXPECTED]
            if len(matches)!=1:
                raise ValueError('Ambiguous category: '+item.filename)
            label=matches[0]; counts[label]+=1
            dest=out/label/(hashlib.sha256(item.filename.encode('utf8')).hexdigest()[:20]+'.jpg')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with z.open(item) as src, Image.open(src) as image:
                    image=ImageOps.exif_transpose(image).convert('RGB')
                    # Deterministic square resize; training augmentation is applied separately.
                    image.resize((256,256), Image.Resampling.LANCZOS).save(dest,format='JPEG',quality=95)
            manifest.append({'archive_path':item.filename,'cache_path':str(dest.relative_to(ROOT)), 'label':label})
            if len(manifest)%100==0: print('prepared',len(manifest),'images',flush=True)
    if counts!=EXPECTED:
        raise ValueError('Unexpected counts: %s; expected %s. Inspect archive before training.'%(counts,EXPECTED))
    (out/'manifest.json').write_text(json.dumps({'source_md5':MD5,'resize':[256,256],'counts':counts,'images':manifest},ensure_ascii=False,indent=2),encoding='utf8')
    print('complete',counts,flush=True)

if __name__=='__main__': main()

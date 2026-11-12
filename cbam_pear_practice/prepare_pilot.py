"""Fetch a balanced exploratory pilot by HTTP ranges from the official DiaMOS archive."""
import concurrent.futures, hashlib, io, json, random, struct, time, zipfile, zlib
from pathlib import Path, PurePosixPath
import requests
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parent
URL='https://zenodo.org/records/5557313/files/Pear.zip?download=1'
SIZE=13122401949
EXPECTED={'healthy':43,'spot':884,'curl':54,'slug':2025}
class RemoteFile(io.RawIOBase):
 def __init__(self): self.pos=0; self.session=requests.Session()
 def seekable(self): return True
 def readable(self): return True
 def tell(self): return self.pos
 def seek(self,offset,whence=0):
  if whence==0:self.pos=offset
  elif whence==1:self.pos+=offset
  elif whence==2:self.pos=SIZE+offset
  return self.pos
 def read(self,n=-1):
  if n<0:n=SIZE-self.pos
  if n<=0:return b''
  data=fetch(self.pos,min(SIZE-1,self.pos+n-1)); self.pos+=len(data); return data

def fetch(start,end):
 for attempt in range(6):
  try:
   r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=120)
   if r.status_code in (429,500,502,503,504):
    time.sleep(min(30,3*(attempt+1))); continue
   r.raise_for_status()
   if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
    raise ValueError('Wrong Range response '+str((r.status_code,r.headers.get('Content-Range'))))
   return r.content
  except (requests.RequestException,ValueError):
   if attempt==5: raise
   time.sleep(2*(attempt+1))
 raise RuntimeError('Range retry limit exceeded')

def convert(record):
 name, label, offset, compressed, original_size, method, crc=record
 end=min(SIZE-1,offset+30+65535+65535+compressed-1)
 # Local ZIP extra fields and names in this archive are short; one extra 4 KiB is ample.
 end=min(SIZE-1,offset+30+4096+compressed-1)
 payload=fetch(offset,end)
 if payload[:4]!=b'PK\x03\x04':raise ValueError('bad local header '+name)
 name_len,extra_len=struct.unpack_from('<HH',payload,26)
 start=30+name_len+extra_len
 if start+compressed>len(payload):
  payload=fetch(offset,min(SIZE-1,offset+start+compressed-1))
 raw=payload[start:start+compressed]
 if method==8: raw=zlib.decompress(raw,-15)
 elif method!=0: raise ValueError('Unsupported compression method')
 if len(raw)!=original_size or (zlib.crc32(raw)&0xffffffff)!=crc: raise ValueError('ZIP CRC mismatch '+name)
 dest=ROOT/'data'/'pilot_256'/label/(hashlib.sha256(name.encode('utf8')).hexdigest()[:20]+'.jpg')
 dest.parent.mkdir(parents=True,exist_ok=True)
 with Image.open(io.BytesIO(raw)) as im:
  im=ImageOps.exif_transpose(im).convert('RGB')
  im.resize((256,256),Image.Resampling.LANCZOS).save(dest,'JPEG',quality=95)
 return {'archive_path':name,'cache_path':str(dest.relative_to(ROOT)),'label':label}

def main():
 groups={k:[] for k in EXPECTED}
 with zipfile.ZipFile(RemoteFile()) as z:
  for i in z.infolist():
   path=PurePosixPath(i.filename.replace('\\','/')); parts=[x.lower() for x in path.parts]
   if i.is_dir() or path.name.startswith('._') or '__macosx' in parts or 'leaves' not in parts or path.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
   labels=[x for x in parts[parts.index('leaves')+1:-1] if x in EXPECTED]
   if len(labels)!=1:raise ValueError('Ambiguous category '+i.filename)
   groups[labels[0]].append((i.filename,labels[0],i.header_offset,i.compress_size,i.file_size,i.compress_type,i.CRC))
 counts={k:len(v) for k,v in groups.items()}
 if counts!=EXPECTED:raise ValueError('Unexpected archive class counts: '+str(counts))
 rng=random.Random(42); chosen=[]
 for label in ('healthy','spot','curl','slug'): chosen+=rng.sample(sorted(groups[label]),40)
 print('Official central-directory counts',counts,'pilot: 40 images per class',flush=True)
 result=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for item in pool.map(convert,chosen):
   result.append(item)
   if len(result)%10==0:print('pilot prepared',len(result),'/',len(chosen),flush=True)
 out=ROOT/'data'/'pilot_256'/'manifest.json'
 out.write_text(json.dumps({'source':URL,'source_bytes':SIZE,'selection':'random 40 per class, seed 42','counts':{x:40 for x in EXPECTED},'images':result},ensure_ascii=False,indent=2),encoding='utf8')
 print('Pilot complete',len(result),flush=True)
if __name__=='__main__':main()

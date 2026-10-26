"""Full-document fixed-length pretraining, dropping each document's tail."""
import hashlib,json
from pathlib import Path
ROOT=Path('/shared-volume/minimind-full-pretrain-20261024')
SOURCE=Path('/shared-volume/fineweb-edu-10BT-score-3.5')
RUNTIME=Path('/tmp/minimind-full-pretrain')
LENGTH=340
STRIDE=LENGTH-1
PROJECT='MiniMind-FineWeb-Full-Pretrain'
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def split_for(text):
 # Identical text always belongs to the same split, independent of document order.
 return 'validation' if int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%4096==0 else 'train'
def chunk_count(length):return max(0,(length-1)//STRIDE)
def full_chunks(ids):
 for i in range(chunk_count(len(ids))):yield ids[i*STRIDE:i*STRIDE+LENGTH]

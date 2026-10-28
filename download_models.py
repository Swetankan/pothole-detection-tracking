"""Download selected model files and verify the hashes used in the delivered run."""
import hashlib,json,urllib.request
from pathlib import Path
for m in json.loads(Path('models_manifest.json').read_text()):
 p=Path(m['path']);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']:
  print(p,'already verified');continue
 tmp=p.with_suffix('.download')
 try:
  urllib.request.urlretrieve(m['url'],tmp)
  if hashlib.sha256(tmp.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Model checksum changed: '+str(p))
  tmp.replace(p);print(p,'downloaded and verified')
 finally:
  if tmp.exists():tmp.unlink()

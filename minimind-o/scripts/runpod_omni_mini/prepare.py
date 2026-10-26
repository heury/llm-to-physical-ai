"""Download immutable sources; no credentials are embedded in artifacts."""
import json, os, time
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
ROOT=Path(os.environ.get('OMNI_RUN_ROOT','/workspace/omni-mini-20261025'))
REVISIONS={
 'jingyaogong/minimind-o_dataset':'d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de',
 'caspar/MiniMind-V-LLaVA-Short-Training-v1.0':'8232a8bf64fb685b89bdf6fc979ab79447f43655',
 'jingyaogong/minimind-3o-pytorch':'9e22fb0e51852359ff51bd48d91ec3a345dbd75b',
 'jingyaogong/SenseVoiceSmall':'964f07d8f0e48309a5f37770ac9b5ed2cdca6b91',
 'google/siglip2-base-patch32-256':'94dffa8cb1179de3e03f091dbc3917e5d5a9ae84',
 'kyutai/mimi':'89091b3e466eb6a9d11e537bf26b144f194978f7',
}
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 (ROOT/'revisions.json').write_text(json.dumps(REVISIONS,indent=2))
 for repo,folder,patterns,kind in [
  ('jingyaogong/minimind-3o-pytorch','base',['sft_omni_768.pth','README.md'],'model'),
  ('jingyaogong/SenseVoiceSmall','models/audio',None,'model'),
  ('google/siglip2-base-patch32-256','models/vision',['*.json','*.safetensors'],'model'),
  ('kyutai/mimi','models/mimi',['*.json','*.safetensors'],'model'),
  ('jingyaogong/minimind-o_dataset','data/audio',['*mini.parquet','README.md'],'dataset'),
  ('caspar/MiniMind-V-LLaVA-Short-Training-v1.0','data/vision',['conversations/*','images/*','README.md','LICENSE_NOTES.md','SHA256SUMS.json','provenance/source-cards/*'],'dataset'),
 ]:
  print('DOWNLOAD',repo,flush=True)
  snapshot_download(repo,repo_type=kind,revision=REVISIONS[repo],local_dir=ROOT/folder,allow_patterns=patterns,max_workers=4)
 (ROOT/'PREPARED').write_text(str(time.time()))
if __name__=='__main__':main()

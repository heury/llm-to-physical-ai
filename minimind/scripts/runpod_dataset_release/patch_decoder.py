"""Metadata-only vision v1.0.1: record the validated pixel reproduction environment."""
import json,time
from huggingface_hub import HfApi,CommitOperationAdd
from cards import RELEASE,REPOS
from publish import sha,verify,write
root=RELEASE/'datasets/vision';repo=REPOS['vision'];receipt=json.loads((RELEASE/'vision-published.json').read_text())
assert receipt['tag']=='v1.0'
validation=json.loads((RELEASE/'loading-validation.json').read_text())['vision']
assert validation['image_decoder']=={'Pillow':'11.3.0','libjpeg_turbo':'3.1.1'}
assert len(validation['original_image_and_training_pixel_hash_checks'])==5
notice='''\n## Image decoding reproducibility (v1.0.1 metadata update)\n\nThe original image bytes are unchanged from v1.0. For exact comparison with the stored training pixel hashes, the release validation reproduced all five sampled split checks using **Pillow 11.3.0 with libjpeg-turbo 3.1.1**. Pillow 13.0.0 with libjpeg-turbo 3.2.0 produced a different resized pixel hash for the JPEG `000000562394.jpg`, despite identical original bytes. Thus pixel-level hashes can depend on the decoder build.\n\n`requirements-image-validation.txt` pins the validated Pillow version. Confirm `PIL.features.version_feature("libjpeg_turbo") == "3.1.1"` for bitwise checks; non-wheel or platform-specific builds may differ. This is a verified reproduction environment, **not a claim that the unrecorded original training package version was 11.3.0**. The original archive byte hashes remain the decoder-independent integrity check. Details are in `provenance/image-decoder-validation.json`.\n\nVersion v1.0.1 only adds these reproducibility notes and requirements; image, text, split and token-array data are unchanged. Original v1.0 remains available as an immutable tag.\n'''
p=root/'README.md';text=p.read_text().replace('# MiniMind-V LLaVA Short Training — v1.0','# MiniMind-V LLaVA Short Training — v1.0.1').replace('revision="v1.0"','revision="v1.0.1"').replace('Version: **v1.0**.','Version: **v1.0.1**.')
p.write_text(text+notice)
(root/'requirements-image-validation.txt').write_text('Pillow==11.3.0\n')
write(root/'provenance/image-decoder-validation.json',{'original_data_commit':receipt['commit'],'sample_validation':validation,'comparison':{'Pillow':'13.0.0','libjpeg_turbo':'3.2.0','sample':'sft/000000562394.jpg','original_bytes_match':True,'training_pixel_hash_matches':False},'training_package_version_recorded':False,'data_changed':False})
manifest=json.loads((root/'SHA256SUMS.json').read_text())
changed=['README.md','requirements-image-validation.txt','provenance/image-decoder-validation.json']
for name in changed:manifest[name]={'bytes':(root/name).stat().st_size,'sha256':sha(root/name)}
write(root/'SHA256SUMS.json',manifest);changed.append('SHA256SUMS.json')
api=HfApi();result=api.create_commit(repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=root/n) for n in changed],commit_message='Document verified image decoder environment (v1.0.1; data unchanged)')
complete=dict(manifest);complete['SHA256SUMS.json']={'bytes':(root/'SHA256SUMS.json').stat().st_size,'sha256':sha(root/'SHA256SUMS.json')}
checks=verify(api,repo,result.oid,complete,root)
api.create_tag(repo,tag='v1.0.1',revision=result.oid,repo_type='dataset')
assert HfApi(token=False).dataset_info(repo,revision='v1.0.1').sha==result.oid
receipt.update(original_v1_0_commit=receipt['commit'],commit=result.oid,tag='v1.0.1',verification=checks,verified_at=time.time(),bytes=sum(x['bytes'] for x in complete.values()),metadata_only_patch=True)
write(RELEASE/'vision-published.json',receipt)
print(json.dumps({'repo':repo,'tag':'v1.0.1','commit':result.oid,'verified':True}),flush=True)

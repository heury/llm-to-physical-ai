import json,urllib.request,zipfile,hashlib
from pathlib import Path
def export():
    base=Path('/shared-volume/minimind-v-llava-short-20261025')
    out=Path('/tmp/dataset-release/vision/provenance');out.mkdir(parents=True,exist_ok=True)
    url='https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/70f9d1e5e1a697fe35830875cfc7de1dd590d727/blip_laion_cc_sbu_558k_meta.json'
    p=Path('/tmp/llava_image_metadata.json');urllib.request.urlretrieve(url,p)
    meta=json.loads(p.read_text())
    selected=json.loads((base/'raw/pretrain.json').read_text());keys={r['image'] for r in selected}
    rows=[r for r in meta if r['image'] in keys]
    assert len(rows)==len(keys)==50000
    (out/'pretrain-image-attribution.json').write_text(json.dumps(rows,ensure_ascii=False))
    url2='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
    zpath=Path('/tmp/coco_annotations.zip');urllib.request.urlretrieve(url2,zpath)
    with zipfile.ZipFile(zpath) as z:data=json.loads(z.read('annotations/instances_train2017.json'))
    keys={r['image'] for r in json.loads((base/'raw/sft.json').read_text())}
    licenses={r['id']:r for r in data['licenses']};selected=[dict(r,license_details=licenses[r['license']]) for r in data['images'] if r['file_name'] in keys]
    assert len(selected)==len(keys)
    (out/'coco-image-attribution.json').write_text(json.dumps(selected,ensure_ascii=False))
    (out/'image-attribution-sources.json').write_text(json.dumps({'pretrain':url,'coco':url2,'coco_metadata_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest()},indent=2))
    print('coco_selected',len(selected),flush=True)

if __name__=="__main__":export()

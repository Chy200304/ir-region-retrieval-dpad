"""Fetch the exact public dataset archive used in this experiment."""
import hashlib
import json
from pathlib import Path
import zipfile

SHA256 = '61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e'
ROOT = Path(__file__).resolve().parent

def write_thumbnails(dataset):
    from PIL import Image
    records = json.loads((ROOT/'data/metadata.json').read_text())['records']
    target = ROOT/'data/thumbnails'
    target.mkdir(parents=True, exist_ok=True)
    for record in records:
        with Image.open(dataset/'images/train2017'/f'{record["image_id"]}.jpg') as source:
            crop = source.convert('RGB').crop(record['bbox_xyxy'])
        crop.thumbnail((160,160))
        crop.save(target/f'{record["id"]}.jpg')
    print(f'Prepared {len(records)} local thumbnails')

def main():
    from huggingface_hub import hf_hub_download
    archive = Path(hf_hub_download('LibreYOLO/coco128', 'coco128.zip',
                                 repo_type='dataset', local_dir=ROOT/'data/download'))
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError(f'Dataset changed: expected {SHA256}, got {actual}')
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(ROOT/'data')
    write_thumbnails(ROOT/'data/coco128')
    print('Verified dataset at', ROOT/'data/coco128')

if __name__ == '__main__':
    main()

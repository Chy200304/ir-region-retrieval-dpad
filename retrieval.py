"""Small, frozen-CLIP region retrieval experiment inspired by DPAD.

This is an inference-time retrieval extension, not a DPAD training reproduction.
"""
import argparse
import hashlib
import html
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
NAMES = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,'
         'fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,'
         'elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,'
         'skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,'
         'surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,'
         'apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,'
         'potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,'
         'cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,'
         'scissors,teddy bear,hair drier,toothbrush').split(',')


def rank(scores):
    # Stable ties make the full-image baseline deterministic.
    return np.argsort(-np.asarray(scores), kind='stable')


def metrics(order, relevant, k=10):
    relevant = np.asarray(relevant, dtype=bool)
    if not relevant.any():
        raise ValueError('Query has no annotated relevant candidates')
    hits = relevant[np.asarray(order)]
    first = np.flatnonzero(hits)[0] + 1
    top = hits[:k]
    ap = float(np.sum(np.cumsum(top) / np.arange(1, len(top) + 1) * top))
    return {'Hit@1': float(hits[0]), 'Hit@10': float(top.any()),
            'Recall@10': float(top.sum() / relevant.sum()),
            'AP@10': ap / min(k, int(relevant.sum())), 'MRR': 1 / float(first)}


def score(crop, full, weight):
    return crop - weight * full


def load_model(cache=None):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    # CPU is the reproducible default; no accelerator is required.
    torch.set_num_threads(4)
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir=cache,
                                    revision=MODEL_REVISION, use_safetensors=False)
    model.eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir=cache,
                                            revision=MODEL_REVISION)
    return torch, model, processor


def encode(torch, model, processor, *, images=None, texts=None):
    chunks = []
    values = images if images is not None else texts
    with torch.inference_mode():
        for start in range(0, len(values), 16):
            batch = values[start:start + 16]
            if images is not None:
                inputs = processor(images=batch, return_tensors='pt')
                features = model.get_image_features(**inputs)
            else:
                inputs = processor(text=batch, padding=True, truncation=True, return_tensors='pt')
                features = model.get_text_features(**inputs)
            # Transformers releases differ in whether feature helpers return tensors.
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().numpy())
    return np.concatenate(chunks)


def build(args):
    from PIL import Image
    source = Path(args.dataset).resolve()
    image_dir, label_dir = source / 'images/train2017', source / 'labels/train2017'
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError('Expected coco128/images/train2017 and labels/train2017')
    started = time.perf_counter()
    records, images, crops, image_ids = [], [], [], []
    thumbs = ROOT / 'data/thumbnails'
    thumbs.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(image_dir.glob('*.jpg')):
        image = Image.open(image_path).convert('RGB')
        parent = len(images)
        images.append(image)
        image_ids.append(image_path.stem)
        label_path = label_dir / (image_path.stem + '.txt')
        # COCO128 omits label files for images without annotated objects.
        lines = label_path.read_text().splitlines() if label_path.exists() else []
        for line_no, line in enumerate(lines, 1):
            category, cx, cy, w, h = map(float, line.split())
            x1, y1 = max(0, int((cx-w/2)*image.width)), max(0, int((cy-h/2)*image.height))
            x2, y2 = min(image.width, int((cx+w/2)*image.width)), min(image.height, int((cy+h/2)*image.height))
            if x2-x1 < 8 or y2-y1 < 8:
                continue
            crop = image.crop((x1, y1, x2, y2))
            identifier = f'{image_path.stem}_{line_no}'
            preview = crop.copy()
            preview.thumbnail((160, 160))
            preview.save(thumbs / f'{identifier}.jpg')
            crops.append(crop)
            records.append({'id': identifier, 'image_id': image_path.stem, 'parent': parent,
                            'category': int(category), 'bbox_xyxy': [x1,y1,x2,y2]})
    torch, model, processor = load_model(args.cache)
    image_features = encode(torch, model, processor, images=images)
    crop_features = encode(torch, model, processor, images=crops)
    queries = [f'a photo of a {name}' for name in NAMES]
    text_features = encode(torch, model, processor, texts=queries)
    np.savez_compressed(ROOT / 'data/features.npz', crops=crop_features,
                        images=image_features, texts=text_features)
    metadata = {'records': records, 'image_ids': image_ids, 'queries': queries,
                'model': 'openai/clip-vit-base-patch32', 'model_revision': MODEL_REVISION,
                'torch': torch.__version__,
                'platform': platform.platform(), 'device': 'cpu',
                'build_seconds': time.perf_counter()-started,
                'protocol': 'All annotated boxes at least 8px in both dimensions; no fitting; fixed weights.'}
    (ROOT / 'data/metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('records','image_ids','queries')}, indent=2))


def evaluate(args):
    data = np.load(ROOT / 'data/features.npz')
    meta = json.loads((ROOT / 'data/metadata.json').read_text())
    records = meta['records']
    parents = np.array([r['parent'] for r in records])
    categories = np.array([r['category'] for r in records])
    crop = data['texts'] @ data['crops'].T
    full = data['texts'] @ data['images'].T
    methods = {'full': full[:,parents], 'crop': crop,
               **{f'contrast_{w}': score(crop, full[:,parents], w) for w in (0.25,0.5,1.0)}}
    present = sorted(set(categories.tolist()))
    per_query, summary = [], {}
    for method, scores in methods.items():
        values = []
        for category in present:
            value = metrics(rank(scores[category]), categories == category)
            values.append(value)
            per_query.append({'method':method, 'category':category, 'query':NAMES[category], **value})
        summary[method] = {key:float(np.mean([v[key] for v in values])) for key in values[0]}
    # Paired category bootstrap is diagnostic, not a test of independent user queries.
    rng = np.random.default_rng(20261106)
    baseline = [r['AP@10'] for r in per_query if r['method']=='crop']
    contrast = [r['AP@10'] for r in per_query if r['method']=='contrast_0.5']
    delta = np.array(contrast)-baseline
    boot = np.mean(delta[rng.integers(0,len(delta),(2000,len(delta)))], axis=1)
    result = {'n_images':len(meta['image_ids']), 'n_regions':len(records),
              'n_queries':len(present), 'summary':summary,
              'contrast_0.5_minus_crop_AP10': {'mean':float(delta.mean()),
                  'bootstrap95':np.quantile(boot,[0.025,0.975]).tolist()},
              'features_sha256':hashlib.sha256((ROOT/'data/features.npz').read_bytes()).hexdigest()}
    (ROOT/'results/metrics.json').write_text(json.dumps(result,indent=2))
    (ROOT/'results/per_query.json').write_text(json.dumps(per_query,indent=2))
    print(json.dumps(result,indent=2))


def demo(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlsplit
    import base64
    data = np.load(ROOT/'data/features.npz')
    meta = json.loads((ROOT/'data/metadata.json').read_text())
    categories = sorted({r['category'] for r in meta['records']})
    parents = np.array([r['parent'] for r in meta['records']])
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = parse_qs(urlsplit(self.path).query)
            try:
                category = int(params.get('category',[categories[0]])[0])
                if category not in categories:
                    raise ValueError()
            except ValueError:
                self.send_error(400, 'Choose a category in the list')
                return
            text = data['texts'][category]
            crop = data['crops'] @ text
            full = (data['images'] @ text)[parents]
            options = ''.join(f'<option value="{c}" {"selected" if c==category else ""}>{html.escape(NAMES[c])}</option>' for c in categories)
            panels = []
            for title,scores in [('Crop cosine',crop),('Crop minus 0.5 full-image cosine',score(crop,full,0.5))]:
                cards = []
                for position in rank(scores)[:10]:
                    r=meta['records'][position]
                    picture=base64.b64encode((ROOT/f'data/thumbnails/{r["id"]}.jpg').read_bytes()).decode()
                    cards.append(f'<article><img src="data:image/jpeg;base64,{picture}"><p>{html.escape(NAMES[r["category"]])} · {scores[position]:.4f}</p><small>{r["id"]}</small></article>')
                panels.append(f'<h2>{title}</h2><div class="grid">'+''.join(cards)+'</div>')
            body=('<!doctype html><meta charset="utf-8"><title>Region retrieval</title><style>body{font:16px system-ui;margin:32px;max-width:1200px;background:#fafafa}select,button{font:inherit;padding:8px}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}article{padding:12px;background:white;border:1px solid #ddd}img{width:100%;height:140px;object-fit:contain}small{font-size:10px}</style><h1>Text to region retrieval</h1><p>Frozen CLIP · annotated candidate boxes · class-name queries · no training</p><form><label>Query category <select name="category">'+options+'</select></label> <button>Search</button></form>'+''.join(panels)).encode()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    print(f'Open http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build'); p.add_argument('--dataset',required=True); p.add_argument('--cache'); p.set_defaults(func=build)
    p=sub.add_parser('evaluate'); p.set_defaults(func=evaluate)
    p=sub.add_parser('demo'); p.add_argument('--port',type=int,default=18765); p.set_defaults(func=demo)
    args=parser.parse_args(); args.func(args)

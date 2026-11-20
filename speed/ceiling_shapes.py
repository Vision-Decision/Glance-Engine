"""Record useful (unpadded) work for the exact benchmark calls, without loading model weights."""
import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from dv.config import Config
from dv.data.dataset import resize_dims
from dv.data.mixed import ChatTok
from scan.data import QUERIES
from speed import workload as W


def collect(out):
    tok = ChatTok(AutoProcessor.from_pretrained('Qwen/Qwen3.5-4B'), Config())
    prefix = len(tok.user_open) + 2

    def shape(image, qs):
        lengths = tuple(len(tok.box_block(q['text']) if q['type'] == 'box' else tok.typed_block(q)[0]) for q in qs)
        return image['visual'], image['prefix'], lengths

    def image(wh, budget, exprs, upscale=2.0):
        w, h = resize_dims(*wh, budget, upscale)
        visual = w*h//1024
        return dict(visual=visual, prefix=prefix+visual, exprs=exprs)

    def calls(pool, b, q, kind, nc, seed=0):
        bursts = []
        for c in range(nc):
            ids = [(c*b+j) % len(pool) for j in range(b)]
            bursts.append(compact([shape(pool[i], W.questions(pool, i, q, kind, seed)) for i in ids]))
        return bursts

    def compact(shapes):
        return [[n, v, p, qs] for (v, p, qs), n in Counter(shapes).items()]

    source = [json.loads(s) for s in Path('data/prep/eval_refcoco_val.jsonl').read_text().splitlines()]
    sizes, exprs = [], []
    for r in source:
        path = Path('/content/data/coco/train2014') / r['file']
        if path.exists():
            with Image.open(path) as im:
                sizes.append(im.size)
            exprs.append([q['text'] for q in r['questions']])
        if len(sizes) == 256:
            break
    assert len(sizes) == 256
    pools = {tv: [image(wh, tv, ex) for wh, ex in zip(sizes[:100], exprs[:100])] for tv in (256, 640)}
    result = {}
    for q in (1, 8, 32):
        for kind in ('box', 'mixed'):
            if q > 1 or kind == 'box':
                result[f'bs1_{kind}_N{q}'] = calls(pools[640], 1, q, kind, 50)
            result[f'followup_{kind}_N{q}'] = calls(pools[640], 1, q, kind, 50, 1)
    for n in (1, 4, 16, 64):
        result[f'clients_{n}'] = calls(pools[640], n, 1, 'box', 12)
    for n in (32, 64):
        result[f'video_f{n}_N1'] = calls(pools[256], n, 1, 'box', 12)
    gui = [json.loads(s) for s in Path('/content/gui/sspro40.jsonl').read_text().splitlines()]
    for tv in (2048, 4096):
        pool = []
        for r in gui[:40]:
            with Image.open(r['path']) as im:
                pool.append(image(im.size, tv, [q['text'] for q in r['questions']], 1.0))
        for q in (1, 8, 32):
            result[f'gui{tv}_box_N{q}'] = calls(pool, 1, q, 'box', 30)
    for tv in (64, 128, 256, 640, 96):
        frame = image((640, 480), tv, [])
        pool = [image(wh, tv, []) for wh in sizes] if tv == 640 else [frame] * 256
        if tv == 96:
            frame['visual'] *= 2
            frame['prefix'] = prefix + frame['visual'] + 2*(len(tok.enc('<0.0 seconds>')) + 2)
        for n in (32, 64, 128, 256):
            for kind in ('noul', 'box'):
                for fresh in (False, True):
                    if fresh and (tv not in (128, 256) or kind != 'noul'):
                        continue
                    if not fresh and (n not in (64, 256) or tv == 128 or
                                      (kind == 'noul' and tv not in (640, 96)) or (kind == 'box' and tv == 96)):
                        continue
                    key = f'fresh_Tv{tv}_N{n}' if fresh else (f'f4_v96_N{n}_noul' if tv == 96 else f'Tv{tv}_N{n}_{kind}')
                    result[key] = []
                    for c in range(10 if fresh else 30):
                        text = QUERIES[c % len(QUERIES)]
                        qs = [dict(type='box', text=text) if kind == 'box' else
                              dict(type='noul', instr=f'Is there {text} in the image?')]
                        result[key].append(compact([shape(im, qs) for im in pool[:n]]))
    with open(out, 'w') as f:
        json.dump(result, f, separators=(',', ':'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    collect(parser.parse_args().out)

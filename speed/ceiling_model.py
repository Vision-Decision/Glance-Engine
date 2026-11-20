"""Compute the C0 useful-work stage floors from measured rates and unpadded call shapes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def call_floor(key, units, rates):
    # Each unit is [multiplicity, visual_tokens, prefix_tokens, question_lengths].
    cached = key.startswith(('Tv', 'f4_', 'followup'))
    bank = key.startswith(('Tv', 'f4_'))
    num = sum(count for count, *_ in units)
    visual = sum(count*v for count, v, _, _ in units)
    patches = 4*visual
    patch_pairs = sum(count*(4*v)**2 for count, v, _, _ in units)
    prefix = sum(count*p for count, _, p, _ in units)
    n_q = sum(count*len(qs) for count, _, _, qs in units)
    q_tokens = sum(count*sum(qs) for count, _, _, qs in units)
    tokens = q_tokens + (0 if cached else prefix)
    pairs = sum(count*(sum(q*(p+q) for q in qs) + (0 if cached else p*(p+1)//2))
                for count, _, p, qs in units)
    stages, resources = {}, defaultdict(float)

    def stage(name, flops, nbytes, dtype='bf16'):
        compute = flops / (rates[dtype]*1e9)
        memory = nbytes / (rates['copy']*1e6)
        stages[name] = max(compute, memory)
        resources[dtype if compute >= memory else 'device BW'] += stages[name]

    if not cached:
        # 2 RGB frames x 16^2 pixels per patch, same FP32 host buffer as both runners.
        stages['H2D'] = patches*1536*4/(rates['h2d']*1e6)
        resources['H2D'] += stages['H2D']
        embed = 1536*1024
        tower = 24*(4*1024**2 + 2*1024*4096)
        merge = 4096*(4096+2560)
        stage('ViT dense', 2*((embed+tower)*patches + merge*visual),
              2*(embed+tower+merge + 2*24*patches*1024))
        stage('ViT attention', 24*4*1024*patch_pairs, 24*2*4*1024*patches)

    d = 2560
    # 32 SwiGLU layers, 24 GDN and 8 gated GQA projection pairs.
    dense = 32*3*d*9216 + 24*d*(2*2048+3*4096+64) + 8*d*(3*4096+2*1024)
    stage('LM dense', 2*dense*tokens, 2*(dense+2*32*d*tokens))
    # One input state and three conv tail entries per distinct cached prefix, ideal reuse across questions.
    state_bytes = num*24*(4*32*128*128 + 2*3*8192) if cached else 0
    stage('GDN', 24*tokens*(6*32*128*128 + 2*4*8192),
          24*2*tokens*(2*2048+2*4096) + state_bytes)
    kv_bytes = 8*2*2*1024*prefix if cached else 0
    stage('LM attention', 8*4*4096*pairs, 8*2*tokens*(2*4096+2*1024) + kv_bytes)
    # Both heads run in current forwards. For clips with no spatial targets the bank holds one dummy key.
    head_v = num if key.startswith('f4_') else visual
    scores = sum(count*(1 if key.startswith('f4_') else v)*len(qs) for count, v, _, qs in units)
    h, bins = 512, 64
    readout = (d*h+h + d*h + (2*d+64)*3*h + 2*h*4*bins+h + d*(30+28))
    wp = 0 if bank else d*h
    stage('heads', 2*(readout*n_q + wp*head_v + h*scores),
          4*(readout+wp + head_v*(h if bank else d) + n_q*d), 'fp32')
    stages['launch'] = rates['launch_ms']
    resources['launch'] += stages['launch']
    return stages, resources


def main(raw):
    peaks = json.loads((raw/'c0_peaks.json').read_text())
    shapes = json.loads((raw/'c0_shapes.json').read_text())
    rates = {k:max(r['tflops'] for r in peaks['gemm'] if r['dtype'] == 'torch.'+dt)
             for k, dt in [('bf16', 'bfloat16'), ('fp32', 'float32')]}
    rates.update(copy=peaks['copy']['GBs'], h2d=peaks['pinned']['GBs'], launch_ms=peaks['replay']['ms'])
    result = {}
    for key, calls in shapes.items():
        measured = [call_floor(key, units, rates) for units in calls]
        stage = {k:median(s[k] for s,_ in measured) for k in measured[0][0]}
        resources = {k:median(r.get(k, 0) for _,r in measured) for k in set().union(*(r for _,r in measured))}
        result[key] = dict(floor_ms=median(sum(s.values()) for s,_ in measured),
                           stages_ms=stage, resource_ms=resources, bound=max(resources, key=resources.get))
    (raw/'c0_floors.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('raw', type=Path)
    main(p.parse_args().raw)

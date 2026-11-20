# README animations

Both GIFs replay measured runs on one NVIDIA RTX PRO 6000 Blackwell (Colab G4), 2026-11-19. Every number and box on screen comes from the JSON timelines in [`data/`](data/). They compare speed; the boxes are each model's real output and are not scored for accuracy here.

## 32 questions about one image ([`glance_vs_vllm.gif`](glance_vs_vllm.gif))

One COCO val2017 photo (#275198) resized to ≤640 visual tokens, 32 RefCOCO-style box questions. Each system is timed over 50 calls; the animation replays the call whose total time is the median, at 10× slow motion. Glance returns all answers from one forward; vLLM (Qwen3.5-4B, greedy, up to 24 new tokens) completes requests one by one.

| Questions | Glance BF16 | vLLM FP8+MTP | vLLM BF16 |
|---:|---:|---:|---:|
| 1 | 37.7 | 81.4 | 159.7 |
| 8 | 40.7 | 139.0 | 227.8 |
| 32 | **49.6** | **242.3** | 348.1 |

Milliseconds, medians of 50 calls. vLLM clears its prefix, multimodal-processor and encoder caches before every call; requests within a call share the image prefix. Glance draws a box only when P(present) > 0.5. [Glance](data/race_glance.json) · [vLLM FP8+MTP](data/race_vllm_fp8mtp.json) · [vLLM BF16](data/race_vllm_bf16.json)

## One question over 256 cached images ([`search_256.gif`](search_256.gif))

256 COCO val2017 photos at ≤640 visual tokens. Both systems have already seen the images: Glance answers from its bank (built once at 120.5 ms and 75.5 MB per image, not animated), and vLLM has warm prefix and encoder caches because each image was asked a different question first. The animation runs in real time.

| Question | Glance BF16 | vLLM FP8+MTP | vLLM BF16 |
|---|---:|---:|---:|
| a giraffe | 95.9 ms | 2,646 ms | 3,057 ms |
| a toaster | 88.8 ms | 2,608 ms | 2,964 ms |

Medians of 30 Glance calls and 5 vLLM calls. The "none of these" verdict uses a max-statistic threshold (α = 0.1) calibrated on scans for 34 COCO categories excluded from the 256 images; the test questions are held-out categories. The guarantee is that, averaged over calibration draws, a scan without the object raises a false "found" at most 10% of the time; it does not bound misses. [Glance](data/search_glance.json) · [vLLM FP8+MTP](data/search_vllm_fp8mtp.json) · [vLLM BF16](data/search_vllm_bf16.json)

## Images

Only COCO val2017 photos licensed CC BY 2.0 or marked "no known copyright restrictions" are used (selected by [`pick_coco.py`](pick_coco.py)). Each image's COCO id, Flickr URL and license are in [`data/credits.csv`](data/credits.csv).

## Reproduce

```bash
bash demo/vm_run.sh pick fetch race search          # on a G4 VM in /content/dv, after scan/vm/setup.sh --vllm
python demo/make_thumbs.py --pick demo/data/coco_pick.json --image_dir /content/demo_imgs --search demo/data/search_glance.json
python demo/render_search.py                        # locally, numpy + pillow
python demo/render_race.py --image path/to/000000275198.jpg
```


#!/usr/bin/env bash
# VM side: all generative-baseline configurations on this GPU, sequentially.  bash speed/vm_serve.sh GPU_TAG [quick]
#   vllm          vLLM, automatic prefix caching, CUDA graphs (defaults)
#   vllm_async    + async scheduling
#   vllm_mtp      + async scheduling + MTP speculative decoding (Qwen3.5's own MTP head, 2 draft tokens)
#   sglang        SGLang, radix prefix cache, CUDA graphs (defaults)
#   sglang_mtp    + MTP speculative decoding (NEXTN / EAGLE worker, 2 steps)
#   vllm_fp8mtp   vllm_mtp with FP8 weights (the FP8-vs-FP8 comparison)
cd /content/dv
G=${1:?gpu tag}; Q=${2:-}
X=""; [ "$Q" = "quick" ] && X="--quick"
[ -n "${NCALLS:-}" ] && X="$X --n_calls $NCALLS"
SET=${SERVE_SET:-vllm,vllm_async,vllm_mtp,sglang,sglang_mtp,vllm_fp8mtp}
L=/content/out/spd/serve_queue_$G.log
for spec in "vllm vllm" "vllm vllm_async --async_sched" "vllm vllm_mtp --async_sched --spec mtp --spec_k 2" \
            "sglang sglang" "sglang sglang_mtp --spec NEXTN --spec_k 2" \
            "vllm vllm_fp8mtp --async_sched --spec mtp --spec_k 2 --quant fp8"; do
  set -- $spec
  BE=$1; TAG=$2; shift 2
  case ",$SET," in *",$TAG,"*) ;; *) continue ;; esac
  echo "[serve] $TAG start $(date +%H:%M:%S)" >> $L
  bash speed/run_serve.sh $BE ${TAG}_$G $X "$@"
  echo "[serve] $TAG end $(date +%H:%M:%S) $(tail -1 /content/out/spd/serve_${TAG}_$G.log)" >> $L
  sleep 5
done
echo "[serve] ALL DONE" >> $L

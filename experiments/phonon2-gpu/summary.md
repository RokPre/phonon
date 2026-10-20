# Phonon-2 GPU feasibility result

**Outcome: real CUDA transcription succeeded on the NVIDIA GeForce GTX 1070 (SM 6.1).** This was a contained local source experiment, not a Fermion-documented install or supported deployment path.

## Runtime and proof

- CUDA runtime: PyTorch `2.7.1+cu126`, CUDA 12.6; official Phonon-2 engine source loaded unchanged from `docker-phonon2/phonon2_cuda_engine.py`.
- Actual model state: all parameters verified on `cuda:0`, dtype `torch.float16`. The script exits on an unavailable/wrong GPU, wrong dtype, or CPU-resident parameter; it cannot silently fall back.
- Synchronized inference profiling recorded **129 CUDA-timed operator entries**, including 216 FP16 GEMM kernel calls (`hgemm_32x32x32_NT_vec`) and 24 CUTLASS FP16 attention calls. `nvidia-smi` attributed **1,404 MiB** to the benchmark process after model load and **1,522 MiB** after inference; PyTorch measured 1,277.98 MiB peak allocated device memory.
- Peak process RSS: CPU runs **1,098.3–1,101.0 MiB**; CUDA process **8,006 MiB** during model loading.
- The GPU had 2,393 MiB free before model load and 909 MiB free after inference (7,196/8,192 MiB used). This leaves limited shared-GPU headroom; availability varies, and existing GPU processes were left untouched.

## Identical-audio benchmark

Input: public OpenAI Whisper `tests/jfk.flac`, converted to mono, 16 kHz float32 WAV, 11.0 s; SHA-256 `b993778cb8e3b1cff6e67e763d5aec2914dfe59da3280baeabc24339edf4f5a0`. The CPU and CUDA paths used that same file. RTF is inference latency divided by audio duration.

| Path | Load and setup | First audio run | Three steady inference runs | Median | Median RTF | Process RSS |
|---|---:|---:|---:|---:|---:|---:|
| CPU (`fermion-research==0.2.12`, 4 threads) | 1.3–2.2 s per invocation (first process: 1.7 s) | 0.728 s | 0.981, 1.111, 1.011 s | **1.011 s** | **0.09191** | 1,098.3–1,101.0 MiB |
| GTX 1070, FP16 | engine load 40.56 s; constructor 46.43 s including 0.73 s warm-up | **0.28577 s** (first 11 s graph capture) | 0.14364, 0.14298, 0.14336 s | **0.14336 s** | **0.01303** | **8,006 MiB peak** |

The CUDA engine resolved the already-cached model in 0.0013 s. Steady-state CUDA inference was **7.05× faster** than the CPU median on this clip. The first audio call is shown separately because it captures the 11-second input’s CUDA graph.

## Transcript and accuracy

Reference: “And so, my fellow Americans: ask not what your country can do for you; ask what you can do for your country.” ([Wikisource transcript](https://en.wikisource.org/wiki/John_F._Kennedy's_Inaugural_Address)). WER ignores case and punctuation.

- CPU: `And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.` — **0.0 WER**.
- GPU: `And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.` — **0.0 WER**.

## Reproduction

```bash
curl --fail --location --retry 3 \
  --output experiments/phonon2-gpu/fixtures/jfk.flac \
  https://raw.githubusercontent.com/openai/whisper/refs/heads/main/tests/jfk.flac
ffmpeg -hide_banner -loglevel error \
  -i experiments/phonon2-gpu/fixtures/jfk.flac \
  -map_metadata -1 -ac 1 -ar 16000 -c:a pcm_f32le \
  experiments/phonon2-gpu/fixtures/jfk-16k-mono.wav
./experiments/phonon2-gpu/setup.sh
./experiments/phonon2-gpu/run.sh
```

`setup.sh` pins the installed runtime dependencies in `requirements.lock`; the scripts route `HF_HOME`, `FERMION_CACHE_DIR`, `PIP_CACHE_DIR`, `TMPDIR`, CUDA, Torch, Triton, and XDG caches under this repository. Full per-run measurements are in [`results.json`](results.json); captured CPU/engine/profiler logs are in [`cpu.log`](cpu.log) and [`gpu.log`](gpu.log), with absolute repository and executable paths redacted.

## Support boundary

The documented image is `ghcr.io/fermionresearch/phonon-cuda:1.0.8`, built with PyTorch 2.9.1+CUDA 12.8. The installed matching host PyTorch 2.9.1+cu128 has no SM 6.1 kernels and fails even a small FP16 CUDA matrix multiply. Docker had only the `runc` runtime, no NVIDIA Container Toolkit, and no cached Phonon image. No image was pulled, no host packages/configuration were changed, and no existing service was stopped or reconfigured. The successful route uses Fermion's unchanged engine source with an older published CUDA wheel; it demonstrates feasibility here but is not a documented or supported Fermion deployment configuration.

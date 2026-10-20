# Phonon-2 GTX 1070 feasibility prototype

This is a contained local feasibility test. It does not change WhisperFlow or run a server.

## Runtime decision

Fermion documents NVIDIA inference through `ghcr.io/fermionresearch/phonon-cuda:1.0.8`, built from `docker-phonon2/` with PyTorch 2.9.1 and CUDA 12.8. The host has Docker Engine 29.7.2 but exposes only the `runc` runtimes, has no `nvidia-container-cli`/`nvidia-ctk`, and has no cached Phonon CUDA image. The installed host PyTorch 2.9.1+cu128 excludes SM 6.1 (`torch.cuda.get_arch_list()` lists SM 7.0 and newer); an FP16 matrix multiply fails with `no kernel image is available for execution on the device`. No image was pulled and no Docker or host configuration was changed.

For feasibility, this prototype runs Fermion's **unchanged** `docker-phonon2/phonon2_cuda_engine.py` source directly in a repository-local environment with the published PyTorch 2.7.1+cu126 wheel. That wheel includes SM 6.0 kernels and passed an FP16 CUDA operation on the GTX 1070. The model engine is explicitly set to `PHONON2_CUDA_DTYPE=float16`; it has no CPU fallback. The environment is an experimental local source run, **not a Fermion-documented or supported deployment path**.

## Setup and run

```bash
./experiments/phonon2-gpu/setup.sh
./experiments/phonon2-gpu/run.sh
```

All Python packages, Hugging Face downloads, Fermion model data, temporary files, and pip cache are placed under this repository in `.venv/` and `.cache/`. Those generated directories are ignored by Git. The setup uses Python 3.12, PyTorch 2.7.1+cu126, Transformers 5.17.0, and `fermion-research` 0.2.12; exact package pins are in `requirements.lock`.

The fixture is OpenAI Whisper's public `tests/jfk.flac` (11 seconds, originally stereo 44.1 kHz). It was converted with:

```bash
ffmpeg -hide_banner -loglevel error -i experiments/phonon2-gpu/fixtures/jfk.flac \
  -map_metadata -1 -ac 1 -ar 16000 -c:a pcm_f32le \
  experiments/phonon2-gpu/fixtures/jfk-16k-mono.wav
```

Both engines read that exact mono 16 kHz WAV. The CPU path is the `fermion-research==0.2.12` CLI pinned to CPU, four threads. The CUDA path imports the repository's upstream CUDA engine source and asserts the GTX 1070, SM 6.1, CUDA-resident FP16 parameters. CUDA is synchronized before and after each timed inference. The run records a first input run (which captures the 11-second input's CUDA graph) plus three steady-state transcriptions, plus a separate untimed PyTorch CUDA profiler run. Loading, the engine's warm-up, and audio inference are timed separately.

## Evidence files

- `results.json` — environment, audio hashes, every run, load/warm-up/inference times, RTF, process RAM, CUDA allocation and profiler events, transcripts, and WER.
- `summary.md` — concise benchmark results, CUDA execution proof, accuracy, and support boundary.
- `cpu.log`, `gpu.log` — captured CLI/engine output, per-process GPU memory, and CUDA profiler table. Absolute repository and executable paths are redacted.
- `fixtures/jfk.flac` and `fixtures/jfk-16k-mono.wav` — public source clip and the shared benchmark input.

The reference phrase is JFK's closing sentence from the [Wikisource transcript](https://en.wikisource.org/wiki/John_F._Kennedy's_Inaugural_Address). WER ignores case and punctuation. The official Docker path remains untested because the required NVIDIA container runtime is not present; this prototype did not modify host services or require external system changes.

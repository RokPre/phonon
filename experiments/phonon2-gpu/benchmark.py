#!/usr/bin/env python3
"""Compare the published Phonon-2 CPU engine with Fermion's unchanged CUDA engine source."""
from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import json
import os
import re
import resource
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIO = HERE / "fixtures" / "jfk-16k-mono.wav"
RUNS = 3
THREADS = 4
REFERENCE = (
    "And so, my fellow Americans: ask not what your country can do for you; "
    "ask what you can do for your country."
)


def local_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name, rel in (
        ("HF_HOME", ".cache/huggingface"),
        ("FERMION_CACHE_DIR", ".cache/fermion"),
        ("PIP_CACHE_DIR", ".cache/pip"),
        ("TMPDIR", ".cache/tmp"),
        ("PHONON_HOME", ".cache/phonon"),
        ("XDG_CACHE_HOME", ".cache/xdg"),
        ("CUDA_CACHE_PATH", ".cache/cuda"),
        ("TORCH_HOME", ".cache/torch"),
        ("TORCH_EXTENSIONS_DIR", ".cache/torch_extensions"),
        ("TORCHINDUCTOR_CACHE_DIR", ".cache/torchinductor"),
        ("TRITON_CACHE_DIR", ".cache/triton"),
    ):
        env[name] = str(ROOT / rel)
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    env.update(
        PHONON2_CUDA_DTYPE="float16",
        PHONON2_CUDA_WARMUP="first",
        PHONON2_CUDA_PAD_S="5",
        OMP_NUM_THREADS=str(THREADS),
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    return env


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(ref) + 1))
    for i, token in enumerate(hyp, 1):
        current = [i]
        for j, expected in enumerate(ref, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (token != expected),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def run_cpu(env: dict[str, str], duration_s: float) -> dict:
    time_bin = Path("/usr/bin/time")
    if not time_bin.is_file():
        raise RuntimeError("GNU time is required to capture per-process CPU RSS")
    cpu_env = env | {"FERMION_DEVICE": "cpu", "FERMION_CPU_THREADS": str(THREADS)}
    command = [
        str(time_bin), "--format=PROCESS_MAX_RSS_KIB=%M",
        str(ROOT / ".venv" / "bin" / "fermion"), "transcribe", "phonon-2",
        str(AUDIO), "--json", "--verbose", "--threads", str(THREADS),
    ]
    entries = []
    with (HERE / "cpu.log").open("w", encoding="utf-8") as log:
        for label in ("first_process_run", *[f"timed_{i}" for i in range(1, RUNS + 1)]):
            start = time.perf_counter()
            result = subprocess.run(
                command, cwd=ROOT, env=cpu_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            wall_s = time.perf_counter() - start
            log.write(f"$ {shlex.join(command[2:])}\n")
            log.write(result.stderr)
            log.write(result.stdout)
            log.write("\n")
            if result.returncode:
                raise RuntimeError(f"CPU run {label} failed ({result.returncode}); see cpu.log")
            payloads = [
                json.loads(line) for line in result.stdout.splitlines()
                if line.lstrip().startswith("{")
            ]
            if not payloads or not payloads[-1].get("text", "").strip():
                raise RuntimeError(f"CPU run {label} produced no transcript; see cpu.log")
            payload = payloads[-1]
            rss = re.search(r"PROCESS_MAX_RSS_KIB=(\d+)", result.stderr)
            loaded = re.search(
                r"loaded .*? on the CPU engine in ([0-9.]+)s", result.stderr
            )
            entries.append({
                "label": label,
                "transcript": payload["text"],
                "load_s": float(loaded.group(1)) if loaded else None,
                "inference_s": float(payload["decode_seconds"]),
                "process_wall_s": round(wall_s, 4),
                "reported_wall_s": payload.get("wall_seconds"),
                "process_peak_rss_mib": round(int(rss.group(1)) / 1024, 2) if rss else None,
                "rtf": round(float(payload["decode_seconds"]) / duration_s, 5),
            })
    steady = entries[1:]
    latencies = [entry["inference_s"] for entry in steady]
    return {
        "engine": "fermion-research 0.2.12 CPU engine",
        "device": "CPU (FERMION_DEVICE=cpu)",
        "threads": THREADS,
        "first_process_run": entries[0],
        "timed_runs": steady,
        "median_inference_s": statistics.median(latencies),
        "median_rtf": statistics.median(latencies) / duration_s,
    }


def smi(query: str) -> str:
    return subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()

def own_gpu_memory_rows() -> list[str]:
    rows = []
    for line in smi("compute-apps=pid,process_name,used_gpu_memory").splitlines():
        fields = line.split(",", 2)
        if len(fields) == 3 and fields[0].strip() == str(os.getpid()):
            rows.append(f"{fields[0].strip()}, benchmark process, {fields[2].strip()}")
    return rows


def run_cuda(audio, duration_s: float) -> dict:
    sys.path.insert(0, str(ROOT / "docker-phonon2"))
    import numpy as np
    import torch
    from entrypoint import resolve_model
    from phonon2_cuda_engine import Transcriber

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    capability = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    if capability != (6, 1) or "GTX 1070" not in device_name:
        raise RuntimeError(f"expected GTX 1070 SM 6.1, found {device_name} SM {capability}")
    if os.environ.get("PHONON2_CUDA_DTYPE") != "float16":
        raise RuntimeError("refusing to run unless PHONON2_CUDA_DTYPE=float16")

    log_path = HERE / "gpu.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"PID={os.getpid()}\n")
        log.write(f"torch={torch.__version__} CUDA={torch.version.cuda}\n")
        log.write(f"device={device_name} capability={capability}\n")
        log.write(f"dtype_request={os.environ['PHONON2_CUDA_DTYPE']}\n")
        log.write("GPU before load:\n" + smi("gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu") + "\n")

        with redirect_stderr(log):
            start = time.perf_counter()
            model_key, model_dir = resolve_model(
                SimpleNamespace(model="phonon-2", model_dir=None, repo=None)
            )
            resolve_s = time.perf_counter() - start
            start = time.perf_counter()
            engine = Transcriber(model_key, model_dir)
            init_wall_s = time.perf_counter() - start
        torch.cuda.synchronize()

        parameters = list(engine.model.parameters())
        devices = sorted({str(parameter.device) for parameter in parameters})
        dtypes = sorted({str(parameter.dtype) for parameter in parameters})
        if not parameters or any(parameter.device.type != "cuda" for parameter in parameters):
            raise RuntimeError(f"model parameters are not all on CUDA: {devices}")
        if any(parameter.dtype != torch.float16 for parameter in parameters):
            raise RuntimeError(f"model parameters are not all FP16: {dtypes}")
        if engine.device_name != device_name or engine.dtype != torch.float16:
            raise RuntimeError("engine did not retain the requested CUDA device and FP16 dtype")

        model_memory_mib = torch.cuda.memory_allocated() / (1024 ** 2)
        process_rows = own_gpu_memory_rows()
        if not process_rows:
            raise RuntimeError("nvidia-smi did not report GPU memory for the benchmark process")
        log.write(f"model_dir={model_dir}\nmodel_resolve_s={resolve_s:.4f}\n")
        log.write(f"engine_load_s={engine.load_s:.4f} engine_init_wall_s={init_wall_s:.4f} warmup_s={engine.warm_s:.4f}\n")
        log.write(f"model_parameter_devices={devices} model_parameter_dtypes={dtypes}\n")
        log.write(f"model_cuda_allocated_mib={model_memory_mib:.2f}\n")
        log.write("nvidia-smi benchmark process memory after load:\n" + "\n".join(process_rows) + "\n")
        torch.cuda.reset_peak_memory_stats()

        def timed_transcription(label: str) -> dict:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            text = engine.transcribe(np.asarray(audio, dtype=np.float32))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if not text.strip():
                raise RuntimeError(f"CUDA inference {label} produced an empty transcript")
            return {
                "label": label,
                "transcript": text,
                "inference_s": round(elapsed, 5),
                "rtf": round(elapsed / duration_s, 5),
            }

        first = timed_transcription("first_audio_run_graph_capture")
        steady = [timed_transcription(f"timed_{i}") for i in range(1, RUNS + 1)]
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        )
        torch.cuda.synchronize()
        with profiler:
            profiled_text = engine.transcribe(np.asarray(audio, dtype=np.float32))
        torch.cuda.synchronize()
        if profiled_text != first["transcript"]:
            raise RuntimeError("profiled CUDA run produced a different transcript")
        events = []
        for event in profiler.key_averages():
            cuda_us = float(getattr(event, "self_device_time_total", 0.0))
            if cuda_us > 0:
                events.append({
                    "op": event.key,
                    "calls": event.count,
                    "self_cuda_us": round(cuda_us, 3),
                })
        events.sort(key=lambda item: item["self_cuda_us"], reverse=True)
        if not events:
            raise RuntimeError("CUDA profiler recorded no CUDA operator time")
        profile_table = profiler.key_averages().table(
            sort_by="self_device_time_total", row_limit=20
        )
        log.write("\nTimed inference runs (each synchronized before and after):\n")
        log.write(json.dumps({"first": first, "steady": steady}, indent=2) + "\n")
        log.write("\nProfiler CUDA operator table (separate untimed run):\n" + profile_table + "\n")
        log.write("\nGPU after inference:\n" + smi("gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu") + "\n")
        process_rows_after = own_gpu_memory_rows()
        if not process_rows_after:
            raise RuntimeError("nvidia-smi no longer reports GPU memory for the benchmark process")
        log.write("nvidia-smi benchmark process memory after inference:\n" + "\n".join(process_rows_after) + "\n")

        latencies = [entry["inference_s"] for entry in steady]
        return {
            "engine": "unchanged docker-phonon2 CUDA engine source",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": device_name,
            "compute_capability": list(capability),
            "dtype": str(engine.dtype).split(".")[-1],
            "model_parameter_devices": devices,
            "model_parameter_dtypes": dtypes,
            "model_resolve_s": round(resolve_s, 4),
            "engine_load_s": round(engine.load_s, 4),
            "engine_init_wall_s": round(init_wall_s, 4),
            "engine_warmup_s": round(engine.warm_s, 4),
            "model_cuda_allocated_mib_after_load": round(model_memory_mib, 2),
            "nvidia_smi_process_rows_after_load": process_rows,
            "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            "first_audio_run": first,
            "timed_runs": steady,
            "median_inference_s": statistics.median(latencies),
            "median_rtf": statistics.median(latencies) / duration_s,
            "profiler": {
                "cuda_operators_with_time": len(events),
                "self_cuda_time_us": round(sum(event["self_cuda_us"] for event in events), 3),
                "top_cuda_operators": events[:20],
            },
        }


def sanitize_logs() -> None:
    for name in ("cpu.log", "gpu.log"):
        path = HERE / name
        text = path.read_text(encoding="utf-8").replace(str(ROOT), "$REPO").rstrip()
        path.write_text(
            "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    env = local_environment()
    os.environ.update(env)
    if not AUDIO.is_file():
        raise SystemExit(f"missing audio fixture: {AUDIO}")
    import soundfile as sf

    audio, sample_rate = sf.read(AUDIO, dtype="float32")
    if sample_rate != 16_000 or audio.ndim != 1 or not len(audio):
        raise SystemExit(f"fixture must be mono 16 kHz audio, got {sample_rate} Hz, shape={audio.shape}")
    duration_s = len(audio) / sample_rate

    cpu = run_cpu(env, duration_s)
    cuda = run_cuda(audio, duration_s)
    sanitize_logs()
    result = {
        "fixture": {
            "original_url": "https://raw.githubusercontent.com/openai/whisper/refs/heads/main/tests/jfk.flac",
            "original_sha256": hashlib.sha256((HERE / "fixtures" / "jfk.flac").read_bytes()).hexdigest(),
            "audio_path": str(AUDIO.relative_to(ROOT)),
            "audio_sha256": hashlib.sha256(AUDIO.read_bytes()).hexdigest(),
            "sample_rate": sample_rate,
            "channels": 1,
            "duration_s": duration_s,
        },
        "reference": {
            "text": REFERENCE,
            "source": "https://en.wikisource.org/wiki/John_F._Kennedy's_Inaugural_Address",
        },
        "cpu": cpu,
        "cuda": cuda,
        "accuracy": {
            "metric": "word error rate; punctuation and case ignored",
            "cpu_wer": word_error_rate(REFERENCE, cpu["timed_runs"][-1]["transcript"]),
            "cuda_wer": word_error_rate(REFERENCE, cuda["timed_runs"][-1]["transcript"]),
        },
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

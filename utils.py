AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"})


def load_config(config_path: str, section: str = "") -> dict:
    """Loads config.yaml and return the full config dict, or a single named section"""
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get(section, {}) if section else cfg
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: could not parse config ({config_path}): {e}")
        return {}


def progress_bar(current: int, total: int, width: int = 28) -> str:
    """Returns an ASCII block progress bar string"""
    if total == 0:
        return "░" * width
    filled = int(width * current / total)
    return "▓" * filled + "░" * (width - filled)


def fmt_time(seconds: float) -> str:
    """Formats duration in seconds as a human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def cuda_available() -> bool:
    """Returns True if a CUDA-capable GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def warn_if_no_cuda() -> None:
    """Prints a warning when CUDA is not available"""
    if not cuda_available():
        print("WARNING: CUDA is not available! Audio processing will run on CPU, which will be much slower.")
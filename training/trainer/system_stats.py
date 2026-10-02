import os
import platform
import subprocess


def rss_mb():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def cpu_count():
    return os.cpu_count() or 1


def hardware_info():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": cpu_count(),
        "rss_peak_mb": rss_mb(),
    }


def git_commit():
    if "RUNE_GIT_COMMIT" in os.environ:
        return os.environ["RUNE_GIT_COMMIT"]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def file_size_bytes(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return -1

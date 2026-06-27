import subprocess

def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def check_ffmpeg() -> bool:
    try:
        run_command(["ffmpeg", "-version"])
        run_command(["ffprobe", "-version"])
        return True
    except Exception:
        return False
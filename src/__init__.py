import os
import sys

# On Windows, add the DLL directories for PyPI-installed nvidia packages (cublas, cudart, cudnn, etc.)
# to the DLL search path so that libraries like ctranslate2 / faster-whisper can find them.
if sys.platform == "win32":
    for package in ["nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn"]:
        try:
            import importlib
            mod = importlib.import_module(package)
            if hasattr(mod, "__file__") and mod.__file__ is not None:
                package_dir = os.path.dirname(mod.__file__)
            elif hasattr(mod, "__path__"):
                package_dir = list(mod.__path__)[0]
            else:
                continue
            bin_dir = os.path.join(package_dir, "bin")
            if os.path.exists(bin_dir):
                os.add_dll_directory(bin_dir)
        except ImportError:
            pass

import re
import subprocess
import sys
from dataclasses import dataclass

# Max CUDA version for each PyTorch version supported by PyG
# Check the list shown on https://data.pyg.org/whl/
# Last updated Oct 2025
TORCH_MAX_CUDA = {
    "2.8": "12.9",
    "2.7": "12.8",
    "2.6": "12.6",
    "2.5": "12.4",
}


@dataclass(frozen=True)
class PyGWheel:
    """The PyG wheel index selected for a Torch/CUDA environment."""

    torch_version: str
    cuda_version: str
    find_links: str
    warning: str | None = None


class UnsupportedTorchVersion(ValueError):
    """A Torch version without a matching PyG extension wheel index."""

    exit_code = 1


class UnsupportedCudaVersion(ValueError):
    """A CUDA version that cannot be mapped to a PyG extension wheel index."""

    exit_code = 1


def _supported_torch_message() -> str:
    latest = max(
        TORCH_MAX_CUDA, key=lambda version: tuple(map(int, version.split(".")))
    )
    return (
        "Unsupported version of PyTorch. Please install a supported version:\n"
        f'  pip install "torch<={latest}"'
    )


def _malformed_cuda_message() -> str:
    return (
        "Unable to determine a valid CUDA version. Please reinstall PyTorch with "
        "a supported CUDA build or use a CPU-only PyTorch build."
    )


def select_pyg_wheel(
    *, torch_version: str, cuda_available: bool, cuda_version: str | None
) -> PyGWheel:
    """Select a PyG extension wheel index without importing Torch or running pip."""
    normalized_torch = torch_version.split("+", maxsplit=1)[0]
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", normalized_torch)
    if match is None:
        raise UnsupportedTorchVersion(_supported_torch_message())

    torch_minor = f"{match.group(1)}.{match.group(2)}"
    if torch_minor not in TORCH_MAX_CUDA:
        raise UnsupportedTorchVersion(_supported_torch_message())

    selected_cuda = "cpu"
    warning = None
    if cuda_available:
        cuda_match = (
            re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", cuda_version)
            if isinstance(cuda_version, str)
            else None
        )
        if cuda_match is None:
            raise UnsupportedCudaVersion(_malformed_cuda_message())

        current_cuda = [int(cuda_match.group(1)), int(cuda_match.group(2))]
        maximum_cuda = list(map(int, TORCH_MAX_CUDA[torch_minor].split(".")))
        if current_cuda[1] < maximum_cuda[1] or (
            current_cuda[0] == maximum_cuda[0] and current_cuda[1] <= maximum_cuda[1]
        ):
            selected_cuda = f"cu{current_cuda[0]}{current_cuda[1]}"
        else:
            warning = (
                "Warning: This CUDA version "
                f"(together with PyTorch {torch_minor}) is unsupported by PyG.\n"
                "\tCheck the list shown on https://data.pyg.org/whl/ for supported "
                "PyTorch-CUDA versions.\n"
                "\tFalling back to the CPU version of CUDA for now..."
            )

    return PyGWheel(
        torch_version=normalized_torch,
        cuda_version=selected_cuda,
        find_links=(
            f"https://data.pyg.org/whl/torch-{normalized_torch}+{selected_cuda}.html"
        ),
        warning=warning,
    )


def main() -> int:
    latest = max(
        TORCH_MAX_CUDA, key=lambda version: tuple(map(int, version.split(".")))
    )

    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Please install PyTorch first:")
        print(f'  pip install "torch<={latest}"')
        return 1

    torch_version_str = torch.__version__.split("+")[0]
    try:
        wheel = select_pyg_wheel(
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
        )
    except (UnsupportedTorchVersion, UnsupportedCudaVersion) as error:
        print(error)
        return error.exit_code

    if wheel.warning:
        print(wheel.warning)

    print(f"Detected PyTorch version: {torch_version_str}")
    print(f"Detected CUDA version: {wheel.cuda_version}")
    print(f"Installing from: {wheel.find_links}")

    # The PyG C++ extensions are only needed by the optional group_averaging
    # module (including torch_scatter imported in group_averaging/model/blocks.py).
    # The core BAM-torch path uses the
    # pure-PyTorch scatter in bam_torch.utils.scatter and does not require any
    # of these extensions. Run this script only if you intend to use
    # `pip install -e ".[group_averaging]"`.
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--find-links",
                wheel.find_links,
                "torch_scatter",
                "torch_cluster",
            ]
        )
        print("Successfully installed torch_scatter and torch_cluster")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print(f"You may need to manually install from: {wheel.find_links}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

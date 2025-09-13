from pathlib import Path
from typing import Tuple

import scipy.io
import torch
from numpy import ndarray as Arr

def _read_mat_file(file_path: Path) -> Tuple[Arr, Arr]:
    """
    Read a MATLAB .mat file and extract the `ArQr` and `AlQl` arrays.

    Args:
        file_path (Path): Path to the .mat file.

    Returns:
        Tuple[Arr, Arr]: A tuple containing the arrays (ArQr, AlQl).

    Raises:
        AssertionError: If the file is not a `.mat` file.
        KeyError: If `ArQr` or `AlQl` keys are missing in the file.
    """
    file_path = Path(file_path)
    assert file_path.suffix == ".mat", "File must be a .mat file"
    mat = scipy.io.loadmat(file_path)
    ArQr, AlQl = mat["ArQr"], mat["AlQl"]
    return ArQr, AlQl

def read_mat_file(d: int, C: int = 25) -> Tuple[Arr, Arr]:
    """
    Load a .mat file for given `d` and `C` values from the default directory.

    Args:
        d (int): Dimension index for the dataset.
        C (int, optional): Constant index (default is 25).

    Returns:
        Tuple[Arr, Arr]: Arrays (ArQr, AlQl) loaded from the file.
    """
    file_pattern = f"FCGram_data_d{d}_C{C}.mat"
    file_path = Path("./FCGram_matrices") / file_pattern
    return _read_mat_file(file_path)

def torch_read_mat_file(d: int, C: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a .mat file and convert its arrays into PyTorch tensors.

    Args:
        d (int): Dimension index for the dataset.
        C (int, optional): Constant index (default is 25).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors (ArQr, AlQl).
    """
    ArQr, AlQl = read_mat_file(d, C)
    return torch.tensor(ArQr), torch.tensor(AlQl)

def convert_all_mat_to_torch(src_dir: Path, dst_dir: Path) -> None:
    """
    Convert all `.mat` files in a directory to `.pt` PyTorch tensor files.

    Args:
        src_dir (Path): Source directory containing .mat files.
        dst_dir (Path): Destination directory to save .pt files.

    Notes:
        - Tensors are saved with dtype `torch.float64` on CPU.
        - Destination directory will be created if it does not exist.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for mat_file in src_dir.glob("FCGram_data_d*_C*.mat"):
        ArQr, AlQl = _read_mat_file(mat_file)
        ArQr_torch = torch.tensor(ArQr, dtype=torch.float64, device="cpu")
        AlQl_torch = torch.tensor(AlQl, dtype=torch.float64, device="cpu")

        dst_file = dst_dir / mat_file.name
        torch.save(
            {
                "ArQr": ArQr_torch,
                "AlQl": AlQl_torch
            },
            dst_file.with_suffix(".pt")
        )

if __name__ == "__main__":
    convert_all_mat_to_torch("./FCGram_matrices", "./FCGram_matrices_torch")
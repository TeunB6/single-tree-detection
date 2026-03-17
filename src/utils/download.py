from pathlib import Path
from src.const import DATA_URL, DATA_PATH
import requests
from zipfile import ZipFile


def download_data(save_path: Path | str, verbose: bool = True):
    save_path = Path(save_path)
    dir = save_path.parent if save_path.is_file() else save_path
    dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading data from {DATA_URL} to {dir}...")
    # Get Zip file
    response = requests.get(DATA_URL)
    if verbose:
        print("Download Status:", response.status_code)
    temp_file = save_path / "temp.zip"

    with open(temp_file, "wb") as f:
        f.write(response.content)

    # Unzip file
    with ZipFile(temp_file, "r") as zip_ref:
        zip_ref.extractall(dir)

    # Remove temp file
    temp_file.unlink()


if __name__ == "__main__":
    download_data(DATA_PATH)

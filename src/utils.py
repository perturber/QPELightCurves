import urllib.request
from pathlib import Path

# Dictionary mapped by custom keys
ARF_URLS = {
    "NICER_simulation": {
        "filename": "nixtiaveonaxis20170601v005.arf",
        "url": "https://heasarc.gsfc.nasa.gov/FTP/caldb/data/nicer/xti/cpf/arf/nixtiaveonaxis20170601v005.arf"
    }
}
def fetch_arf(key: str, target_dir: str = "../downloads") -> Path:
    """
    Fetches a standard ARF by key and returns its local file path.
    Downloads the file if it does not already exist.
    """
    if key not in ARF_URLS:
        raise ValueError(f"Unknown ARF key: '{key}'. Available keys: {list(ARF_URLS.keys())}")
        
    info = ARF_URLS[key]
    download_path = Path(target_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    
    file_path = download_path / info["filename"]
    
    # Check if we already downloaded it
    if file_path.exists():
        print(f"Found local ARF for '{key}': {file_path.absolute()}")
        return file_path
        
    # If not, download it
    print(f"Downloading {info['filename']} for '{key}' to {file_path.absolute()}...")
    req = urllib.request.Request(info["url"], headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        # Clean up the partial file if the download crashes
        if file_path.exists():
            file_path.unlink() 
        raise RuntimeError(f"Failed to download ARF: {e}")
        
    return file_path
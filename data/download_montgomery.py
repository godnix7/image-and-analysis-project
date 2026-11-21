"""Download the NLM Montgomery CXR images and manual lung masks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
import time
import requests

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/"
OUT = Path(__file__).resolve().parent / "MontgomerySet"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def files_in(folder):
    url = urljoin(BASE, folder + "/index.html")
    response = requests.get(url, timeout=40)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    return [urljoin(url, x) for x in parser.hrefs if x.lower().endswith(".png")]


def download(url):
    relative = url.split("/MontgomerySet/", 1)[1]
    dest = OUT / Path(relative)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return relative, dest.stat().st_size
    temp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 6):
        try:
            temp.unlink(missing_ok=True)
            response = requests.get(url, timeout=(30, 180), stream=True)
            response.raise_for_status()
            with temp.open("wb") as f:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        f.write(block)
            temp.replace(dest)
            return relative, dest.stat().st_size
        except requests.RequestException:
            temp.unlink(missing_ok=True)
            if attempt == 5:
                raise
            time.sleep(attempt * 2)


def main():
    folders = ["CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"]
    urls = [url for folder in folders for url in files_in(folder)]
    print(f"Downloading {len(urls)} PNG files from NLM into {OUT}", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, url) for url in urls]
        for future in as_completed(futures):
            name, size = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(urls):
                print(f"{completed}/{len(urls)} files ({size:,} bytes last): {name}", flush=True)
    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()

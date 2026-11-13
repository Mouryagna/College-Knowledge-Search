import os
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
LINKS_FILE = ROOT_DIR / "data" / "Links.txt"
DOWNLOAD_DIR = ROOT_DIR / "data" / "downloads"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}


async def fetch_url(session, url, semaphore):
    async with semaphore:
        # Respect server rate limits with a small sleep pause
        await asyncio.sleep(1.2)
        try:
            async with session.get(url, headers=HEADERS, timeout=20) as response:
                if response.status != 200:
                    print(f"[FAILED {response.status}] {url}")
                    return None

                content_type = response.headers.get("Content-Type", "").lower()
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or "index.html"

                # Check for downloadable binary extensions
                if any(ext in filename.lower() for ext in ['.pdf', '.docx', '.xlsx']) or "application" in content_type:
                    save_path = DOWNLOAD_DIR / filename
                    async with aiofiles.open(save_path, 'wb') as f:
                        await f.write(await response.read())
                    print(f"[DOWNLOADED FILE] {filename}")
                    return {"type": "file", "url": url, "path": str(save_path)}
                else:
                    html_text = await response.text()
                    print(f"[FETCHED HTML] {url}")
                    return {"type": "html", "url": url, "content": html_text}
        except Exception as e:
            print(f"[ERROR] {url}: {e}")
            return None


async def run_crawler():
    if not LINKS_FILE.exists():
        raise FileNotFoundError(f"{LINKS_FILE} not found! Place your links inside data/Links.txt")

    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Starting polite crawl for {len(urls)} URLs (Semaphore=2, Sleep=1.2s)...")
    semaphore = asyncio.Semaphore(2)  # Max 2 concurrent connections to prevent 503

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)

    valid_results = [r for r in results if r is not None]
    print(f"\nCompleted crawl. Total successful items: {len(valid_results)} / {len(urls)}")
    return valid_results


if __name__ == "__main__":
    asyncio.run(run_crawler())
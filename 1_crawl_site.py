import argparse
import logging  # Import the logging module
import os
import shutil

from dpk_web2parquet.transform import Web2Parquet

from my_config import MY_CONFIG

# Configure logging to the WARN level
logging.basicConfig(level=logging.WARN)


def main():
    parser = argparse.ArgumentParser(
        description="Crawl a website and convert it to Parquet format."
    )
    parser.add_argument("--url", type=str, help="URL to crawl", required=True)
    parser.add_argument(
        "--max-downloads",
        type=int,
        help="Maximum number of files to download",
        default=MY_CONFIG.CRAWL_MAX_DOWNLOADS,
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth to crawl",
        default=MY_CONFIG.CRAWL_MAX_DEPTH,
    )

    args = parser.parse_args()

    print(
        f"⚙️  Crawling with URL: {args.url}, Max Downloads: {args.max_downloads}, Max Depth: {args.max_depth}"
    )

    if not os.path.exists(MY_CONFIG.INPUT_DIR):
        shutil.os.makedirs(MY_CONFIG.INPUT_DIR, exist_ok=True)

    # clear input / output folder
    shutil.rmtree(MY_CONFIG.INPUT_DIR, ignore_errors=True)
    shutil.os.makedirs(MY_CONFIG.INPUT_DIR, exist_ok=True)

    print("✅ Cleared  download directory")

    print("⚙️  Starting web crawl...")
    Web2Parquet(
        urls=[args.url],
        depth=args.max_depth,
        downloads=args.max_downloads,
        folder=MY_CONFIG.INPUT_DIR,
    ).transform()

    print(
        f"✅ web crawl completed.  Downloaded {len(os.listdir(MY_CONFIG.INPUT_DIR))} files into '{MY_CONFIG.INPUT_DIR}' directory"
    )


if __name__ == "__main__":
    main()

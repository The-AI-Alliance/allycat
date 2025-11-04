#!/usr/bin/env python3

import argparse
import shutil
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
import os
import re
import mimetypes
from my_config import MY_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebScraper:
    def __init__(self, url, max_downloads, depth):
        self.url = url
        self.max_downloads = max_downloads
        self.depth = depth
        self.visited_urls = set()
        self.downloaded_base_urls = set()  # Track base URLs without fragments
        self.downloaded_count = 0
        
    def scrape_page(self, url, current_depth=0):
        try:
            # For downloading, we need to remove fragment since HTTP requests ignore them
            parsed_url = urlparse(url)
            download_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            if parsed_url.query:
                download_url += f"?{parsed_url.query}"
            
            # Check if we've already downloaded this base URL content
            if download_url in self.downloaded_base_urls:
                # If we have a fragment and haven't visited this exact URL, save with fragment name
                if parsed_url.fragment and url not in self.visited_urls:
                    # Get the cached response if we have one, otherwise make a request
                    response = requests.get(download_url, timeout=10)
                    response.raise_for_status()
                    
                    filename = self.url_to_filename(url, response)
                    filepath = os.path.join(MY_CONFIG.CRAWL_DIR, filename)
                    
                    # Handle binary files vs text files based on mime type
                    mime_type = response.headers.get('Content-Type', '').lower()
                    is_text = mime_type.startswith('text/') or 'html' in mime_type or 'xml' in mime_type
                    
                    if is_text:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(response.text)
                    else:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                    
                    self.downloaded_count += 1
                    logger.info(f"Saved {filepath} with fragment ({self.downloaded_count}/{self.max_downloads})")
                    
                    return []  # Don't re-parse links from same content
                else:
                    logger.info(f"Skipping already downloaded URL: {download_url}")
                    return []
            
            response = requests.get(download_url, timeout=10)
            response.raise_for_status()
            
            # Track that we've downloaded this base URL
            self.downloaded_base_urls.add(download_url)
            
            # Save file using original URL (with fragment) for unique filename
            filename = self.url_to_filename(url, response)
            filepath = os.path.join(MY_CONFIG.CRAWL_DIR, filename)
            
            # Handle binary files vs text files based on mime type
            mime_type = response.headers.get('Content-Type', '').lower()
            is_text = mime_type.startswith('text/') or 'html' in mime_type or 'xml' in mime_type
            
            if is_text:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            else:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            
            self.downloaded_count += 1
            logger.info(f"Saved {filepath} ({self.downloaded_count}/{self.max_downloads})")
            
            # Parse for links if not at max depth
            links = []
            if current_depth < self.depth:
                soup = BeautifulSoup(response.content, 'html.parser')
                base_domain = urlparse(self.url).netloc
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(url, link.get('href'))
                    if urlparse(full_url).netloc == base_domain:
                        links.append(full_url)
            
            return links
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return []
    
    def url_to_filename(self, url, response):
        # Keep domain and path, strip protocol, use __ for directory separators
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        fragment = parsed.fragment
        
        if not path or path == '/':
            filename = f"{domain}__index"
        else:
            filename = f"{domain}{path.replace('/', '__')}"
        
        # Add fragment (anchor) to filename if present
        if fragment:
            filename = f"{filename}__{fragment}"
        
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        
        mime_type = response.headers.get('Content-Type')
        if mime_type:
            inferred_extension = mimetypes.guess_extension(mime_type.split(';')[0].strip())
        else:            
            inferred_extension = '.html'

        current_ext = os.path.splitext(filename)[1]
        ext = os.path.splitext(filename)[1]
        # print ('--- filename:', filename)  # Debugging line
        # print ('--- mimetype:', mime_type)  # Debugging line
        # print ('--- inferred_extension', inferred_extension)  # Debugging line
        # print ('--- current_ext:', current_ext)  # Debugging line

        # Only append .html if no extension exists
        if not filename.endswith(inferred_extension):
             filename = f"{filename}.html"

        # print ('--- returning filename:', filename)  # Debugging line
        return filename
        
    
    def scrape(self):
        shutil.rmtree(MY_CONFIG.CRAWL_DIR, ignore_errors=True)
        os.makedirs(MY_CONFIG.CRAWL_DIR, exist_ok=True)
        logger.info(f"✅ Cleared  crawl directory: {MY_CONFIG.CRAWL_DIR}")

        logger.info(f"⚙ Starting scrape of {self.url}, max downloads: {self.max_downloads}, depth: {self.depth}")

        
        urls_to_visit = [(self.url, 0)]  # (url, depth)
        
        while urls_to_visit and self.downloaded_count < self.max_downloads:
            current_url, current_depth = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            links = self.scrape_page(current_url, current_depth)
            
            # Add new URLs if not at max depth
            if current_depth < self.depth:
                for link in links:
                    if link not in self.visited_urls:
                        urls_to_visit.append((link, current_depth + 1))
            
            time.sleep(MY_CONFIG.WAITTIME_BETWEEN_REQUESTS)
    

def main():
    parser = argparse.ArgumentParser(description="Web scraper")
    parser.add_argument("--url", type=str, required=True, help="URL to scrape")
    parser.add_argument("--max-downloads", type=int, default=MY_CONFIG.CRAWL_MAX_DOWNLOADS, help=f"Maximum number of files to download (default: {MY_CONFIG.CRAWL_MAX_DOWNLOADS})")
    parser.add_argument("--depth", type=int, default=MY_CONFIG.CRAWL_MAX_DEPTH, help=f"Maximum depth to crawl (default: {MY_CONFIG.CRAWL_MAX_DEPTH})")

    args = parser.parse_args()
    
    scraper = WebScraper(args.url, args.max_downloads, args.depth)
    scraper.scrape()
    
    logger.info(f"✅ Scraping completed. Downloaded {scraper.downloaded_count}  files to '{MY_CONFIG.CRAWL_DIR}' directory.")

if __name__ == "__main__":
    main()
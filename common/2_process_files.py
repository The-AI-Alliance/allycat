import os, sys
import shutil
from pathlib import Path
from docling.document_converter import DocumentConverter
import logging
import hashlib
from my_config import MY_CONFIG

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cleanup_duplicate_markdown_files(processed_dir):
    """
    Remove duplicate markdown files based on content hash.
    Keeps the first file encountered for each unique content.
    """
    processed_path = Path(processed_dir)
    md_files = list(processed_path.glob('*.md'))
    
    if not md_files:
        logger.info("No markdown files found for deduplication")
        return 0
    
    content_hashes = {}
    duplicates_removed = 0
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash in content_hashes:
                os.remove(md_file)
                duplicates_removed += 1
                logger.info(f"Removed duplicate: {md_file} (same content as {content_hashes[content_hash]})")
            else:
                content_hashes[content_hash] = md_file
                
        except Exception as e:
            logger.warning(f"Error processing {md_file} for deduplication: {e}")
    
    logger.info(f"✅ Deduplication complete. Removed {duplicates_removed} duplicate files")
    return duplicates_removed
## --- end of cleanup_duplicate_markdown_files ---

def process_files(crawl_dir, processed_dir):
    """
    Process all files in the crawl directory and convert them to markdown.
    
    Args:
        crawl_dir (str): Directory containing files to process
        processed_dir (str): Directory to save processed markdown files
    """

    input_path = Path(crawl_dir)
    input_files = list(input_path.glob('*')) 
    logger.info (f"Found {len(input_files)} files to process in {input_path}")

    shutil.rmtree(processed_dir, ignore_errors=True)
    shutil.os.makedirs(processed_dir, exist_ok=True)
    logger.info (f"✅ Cleared  processed data directory :  {processed_dir}")
    
    converter = DocumentConverter(format_options={"preserve_links": True})
    
    files_processed = 0
    errors = 0
    file_type_stats = {}
    
    for input_file in input_files:
        file_ext = input_file.suffix.lower()
        try:
            result = converter.convert(input_file)
            markdown_content = result.document.export_to_markdown()
            
            md_file_name = os.path.join(processed_dir, f"{input_file.stem}.md")
            with open(md_file_name, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)
                
            logger.debug(f"Converted '{input_file}' --> '{md_file_name}'")
            files_processed += 1
            
            # Track file type statistics
            file_type_stats[file_ext] = file_type_stats.get(file_ext, 0) + 1
            
        except Exception as e:
            errors += 1
            logger.warning(f"Error processing {input_file}: {e}")

    logger.info (f"✅ Processed {files_processed} files.  Errors: {errors}")
    
    # Print file type statistics in compact dictionary format
    if file_type_stats:
        logger.info(f"📊 File type statistics: {dict(sorted(file_type_stats.items()))}")
    
    return files_processed, errors, file_type_stats
## --- end of process_files ---

def main():
    """
    Main function to run the file processing pipeline.
    """
    logger.info("🚀 Starting file processing pipeline")
    
    try:
        files_processed, errors, file_type_stats = process_files(MY_CONFIG.CRAWL_DIR, MY_CONFIG.PROCESSED_DATA_DIR)
        duplicates_removed = cleanup_duplicate_markdown_files(MY_CONFIG.PROCESSED_DATA_DIR)
        logger.info(f"✅ Final summary: {files_processed} files processed, {errors} errors, {duplicates_removed} duplicates removed")
        logger.info("✅ File processing pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"❌ File processing pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
Processing HTML Files using html2parquet transform
References:
- html2parquet: https://github.com/IBM/data-prep-kit/tree/dev/transforms/language/html2parquet/python
"""

import os
import sys
import shutil
from dpk_html2parquet.transform_python import Html2Parquet
from common.file_utils import read_parquet_files_as_df

from my_config import MY_CONFIG

def main():
    PQ_DIR = os.path.join(MY_CONFIG.WORKSPACE_DIR, "parquet")
    shutil.rmtree(PQ_DIR, ignore_errors=True)
    shutil.os.makedirs(PQ_DIR, exist_ok=True)
    print (f"✅ Cleared  intermediate parquet directory :  {PQ_DIR}")

    shutil.rmtree(MY_CONFIG.PROCESSED_DATA_DIR, ignore_errors=True)
    shutil.os.makedirs(MY_CONFIG.PROCESSED_DATA_DIR, exist_ok=True)
    print (f"✅ Cleared  processed data directory :  {MY_CONFIG.PROCESSED_DATA_DIR}")

    Html2Parquet(
        input_folder=MY_CONFIG.CRAWL_DIR,
        output_folder=PQ_DIR,
        data_files_to_use=['.html'],
        html2parquet_output_format="markdown"
    ).transform()

    output_df = read_parquet_files_as_df(PQ_DIR)

    # Step-5: Save the markdown files
    for index, row in output_df.iterrows():
        html_file = row['document']
        base_name = os.path.splitext(os.path.basename(html_file))[0]
        md_output_file = os.path.join(MY_CONFIG.PROCESSED_DATA_DIR, base_name + '.md')
        
        with open(md_output_file, 'w') as md_output_file_handle:
            md_output_file_handle.write(row['contents'])
    
    print(f"✅ Saved {index+1} md files into '{MY_CONFIG.PROCESSED_DATA_DIR}'")

if __name__ == "__main__":
    main()
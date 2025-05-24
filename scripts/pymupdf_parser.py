#!/usr/bin/env python3
"""
Fast PDF to Text Parser using PyMuPDF
This script processes a directory of PDF files and converts them to text using PyMuPDF.
"""

import os
import sys
import argparse
import logging
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.error("PyMuPDF not found. Install with: pip install pymupdf")
    sys.exit(1)


class PyMuPDFParser:
    """
    A class to handle PDF to text conversion using PyMuPDF (fitz)
    """
    
    def __init__(self, output_dir: str = "output_mini", 
                 extraction_mode: str = "text",
                 max_workers: int = 2, 
                 chunk_size: int = 5):
        """
        Initialize the PyMuPDF PDF Parser
        
        Args:
            output_dir: Directory to save the converted text files
            extraction_mode: Text extraction mode (text, blocks, words, html, dict, json, rawdict, xhtml)
            max_workers: Maximum number of parallel workers (defaults to 2)
            chunk_size: Number of PDFs to process in each batch (defaults to 5)
        """
        self.output_dir = Path(output_dir)
        self.extraction_mode = extraction_mode
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        
        # Stats
        self.start_time = None
        self.processed_count = 0
        self.total_count = 0
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_pdf_files(self, directory: str) -> list:
        """
        Find all PDF files in a directory and subdirectories
        
        Args:
            directory: Path to the directory containing PDFs
            
        Returns:
            List of Path objects for PDF files
        """
        pdf_dir = Path(directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")
        
        # Find all PDFs recursively
        pdf_files = list(pdf_dir.glob("**/*.pdf")) + list(pdf_dir.glob("**/*.PDF"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        return pdf_files
    
    def estimate_completion_time(self) -> str:
        """
        Estimate the completion time based on current progress
        
        Returns:
            String with estimated time remaining
        """
        if self.start_time is None or self.processed_count == 0:
            return "Calculating..."
            
        elapsed = time.time() - self.start_time
        files_per_second = self.processed_count / elapsed
        
        if files_per_second > 0:
            remaining_files = self.total_count - self.processed_count
            remaining_seconds = remaining_files / files_per_second
            return str(timedelta(seconds=int(remaining_seconds)))
        else:
            return "Unknown"
    
    def convert_single_pdf(self, pdf_path: Path) -> bool:
        """
        Convert a single PDF file to text using PyMuPDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Check if output already exists and skip if it does
            output_file = self.output_dir / f"{pdf_path.stem}.txt"
            if output_file.exists():
                logger.debug(f"Skipping {pdf_path.name} - output already exists")
                return True
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text based on mode
                if self.extraction_mode == "html":
                    text = page.get_text("html")
                elif self.extraction_mode == "json":
                    text = page.get_text("json")
                elif self.extraction_mode == "xhtml":
                    text = page.get_text("xhtml")
                elif self.extraction_mode == "dict":
                    # For dict mode, we convert to string for saving
                    text = str(page.get_text("dict"))
                elif self.extraction_mode == "rawdict":
                    # For rawdict mode, we convert to string for saving
                    text = str(page.get_text("rawdict"))
                elif self.extraction_mode == "blocks":
                    # For blocks mode, we convert to string for saving
                    blocks = page.get_text("blocks")
                    text = "\n\n".join([f"Block {i}: {block[4]}" for i, block in enumerate(blocks)])
                elif self.extraction_mode == "words":
                    # For words mode, we convert to string for saving
                    words = page.get_text("words")
                    text = " ".join([word[4] for word in words])
                else:
                    # Default to plain text
                    text = page.get_text("text")
                
                full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            # Close the document
            doc.close()
            
            # Save the text to a file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
                
            logger.debug(f"Successfully converted {pdf_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path.name}: {str(e)}")
            return False
    
    def process_in_parallel(self, pdf_files: list) -> tuple:
        """
        Process PDFs in parallel using thread pool
        
        Args:
            pdf_files: List of PDF files to process
            
        Returns:
            Tuple of (successful count, failed count)
        """
        successful = 0
        failed = 0
        self.total_count = len(pdf_files)
        self.processed_count = 0
        self.start_time = time.time()
        
        # Process in chunks for better memory management
        pdf_chunks = [pdf_files[i:i + self.chunk_size] for i in range(0, len(pdf_files), self.chunk_size)]
        logger.info(f"Processing {len(pdf_files)} PDFs in {len(pdf_chunks)} chunks of up to {self.chunk_size} PDFs each")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit all PDFs for processing
            for pdf_file in pdf_files:
                futures.append(executor.submit(self.convert_single_pdf, pdf_file))
                
            # Process results as they complete
            for i, future in enumerate(futures):
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                    
                    self.processed_count += 1
                    completion_pct = (self.processed_count / self.total_count) * 100
                    est_time = self.estimate_completion_time()
                    
                    if self.processed_count % 10 == 0 or self.processed_count == self.total_count:
                        logger.info(f"Progress: {completion_pct:.1f}% ({self.processed_count}/{self.total_count}) - Est. time remaining: {est_time}")
                    
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}")
                    failed += 1
                    self.processed_count += 1
        
        return successful, failed
    
    def convert_directory(self, pdf_directory: str, recompute: bool = False) -> None:
        """
        Convert all PDFs in a directory to text
        
        Args:
            pdf_directory: Path to directory containing PDFs
            recompute: Recompute already processed PDFs
        """
        try:
            pdf_files = self.find_pdf_files(pdf_directory)
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {pdf_directory}")
                return
            
            # Filter out already processed files unless recompute is True
            if not recompute:
                unprocessed_files = []
                for pdf_file in pdf_files:
                    output_file = self.output_dir / f"{pdf_file.stem}.txt"
                    if not output_file.exists():
                        unprocessed_files.append(pdf_file)
                
                if len(unprocessed_files) < len(pdf_files):
                    logger.info(f"Skipping {len(pdf_files) - len(unprocessed_files)} already processed files")
                    pdf_files = unprocessed_files
            
            if not pdf_files:
                logger.info("All files have already been processed")
                return
                
            # Process in parallel
            start_time = time.time()
            successful, failed = self.process_in_parallel(pdf_files)
            elapsed = time.time() - start_time
            
            # Report results
            logger.info(f"Conversion complete! Success: {successful}, Failed: {failed}")
            logger.info(f"Total processing time: {timedelta(seconds=int(elapsed))}")
            logger.info(f"Output files saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Exception during conversion: {str(e)}")


def main():
    """
    Main function to handle command line arguments and run the parser
    """
    parser = argparse.ArgumentParser(description="Convert PDFs to text using PyMuPDF (faster than PyPDF2)")
    parser.add_argument("pdf_dir", nargs='?', default="test_scrap_mini", 
                       help="Directory containing PDF files (default: test_scrap_mini)")
    parser.add_argument("-o", "--output", default="output_mini", 
                       help="Output directory for text files (default: output_mini)")
    parser.add_argument("-m", "--mode", default="text",
                       choices=["text", "html", "json", "dict", "blocks", "words", "rawdict", "xhtml"],
                       help="Text extraction mode (default: text)")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of worker threads (default: 2)")
    parser.add_argument("--chunk-size", type=int, default=5,
                       help="Number of PDFs to process in each batch (default: 5)")
    parser.add_argument("--recompute", action="store_true",
                       help="Reprocess already processed PDFs")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    parser.add_argument("--max-pdfs", type=int, default=None,
                       help="Maximum number of PDFs to process in one run")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Start processing from this index in the PDF list")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize the parser
    pdf_parser = PyMuPDFParser(
        output_dir=args.output,
        extraction_mode=args.mode,
        max_workers=args.workers,
        chunk_size=args.chunk_size
    )
    
    start_time = time.time()
    
    try:
        # If max-pdfs or start-index is specified, use a special mode
        if args.max_pdfs is not None or args.start_index > 0:
            # Find all PDFs
            all_pdfs = pdf_parser.find_pdf_files(args.pdf_dir)
            
            # Apply start index and max count
            start_idx = args.start_index
            end_idx = len(all_pdfs) if args.max_pdfs is None else min(start_idx + args.max_pdfs, len(all_pdfs))
            
            # Get subset of PDFs
            pdfs_to_process = all_pdfs[start_idx:end_idx]
            logger.info(f"Processing PDF subset: {start_idx} to {end_idx-1} (total: {len(pdfs_to_process)})")
            
            # Process this batch
            if len(pdfs_to_process) > 0:
                successful, failed = pdf_parser.process_in_parallel(pdfs_to_process)
                logger.info(f"Batch conversion complete! Success: {successful}, Failed: {failed}")
            else:
                logger.warning("No PDFs to process in the specified range")
        else:
            # Standard processing
            pdf_parser.convert_directory(args.pdf_dir, recompute=args.recompute)
            
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
    
    elapsed = time.time() - start_time
    logger.info(f"Total execution time: {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    # Example usage when running as a script
    if len(sys.argv) == 1:
        print("No arguments provided. Using default settings:")
        print("PDF Directory: test_scrap_mini")
        print("Output Directory: output_mini")
        print("Extraction Mode: text")
        print("Workers: 2")
        print("Chunk Size: 5")
        print()
        print("For processing large directories:")
        print("python pymupdf_parser.py large_pdf_dir -o output_dir --max-pdfs 20 --start-index 0")
        print("python pymupdf_parser.py large_pdf_dir -o output_dir --max-pdfs 20 --start-index 20")
        print()
        print("You can also use different extraction modes:")
        print("python pymupdf_parser.py test_scrap_mini -m html")
        print("python pymupdf_parser.py test_scrap_mini -m blocks")
    
    main()


# Example of how to use this programmatically in another script:
"""
# Import the class
from pymupdf_parser import PyMuPDFParser

# Initialize the parser with default settings
parser = PyMuPDFParser(
    output_dir="output_mini",
    extraction_mode="text",  # or "html", "blocks", "words", etc.
    max_workers=2,
    chunk_size=5
)

# Process all PDFs in a directory
parser.convert_directory("test_scrap_mini")

# Or process a specific batch of PDFs
import os
from pathlib import Path

# Find all PDFs
pdf_dir = "large_pdf_directory"
all_pdfs = list(Path(pdf_dir).glob("**/*.pdf"))

# Process in batches of 20
batch_size = 20
for i in range(0, len(all_pdfs), batch_size):
    batch = all_pdfs[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1} ({i} to {i+len(batch)-1})")
    
    # Process this batch using parallel execution
    successful, failed = parser.process_in_parallel(batch)
    print(f"Batch complete: {successful} succeeded, {failed} failed")
"""
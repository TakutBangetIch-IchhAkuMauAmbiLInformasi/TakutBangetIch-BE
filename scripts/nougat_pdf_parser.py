#!/usr/bin/env python3
"""
PDF to Text Parser using Nougat
This script processes a directory of PDF files and converts them to text using Nougat.
"""

import os
import sys
import argparse
import subprocess
import logging
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NougatPDFParser:
    """
    A class to handle PDF to text conversion using Nougat
    """
    
    def __init__(self, output_dir: str = "output", model: str = "0.1.0-small", 
                 batchsize: int = 1, no_skipping: bool = False, markdown: bool = True,
                 max_workers: int = 2, chunk_size: int = 5):
        """
        Initialize the Nougat PDF Parser
        
        Args:
            output_dir: Directory to save the converted text files
            model: Nougat model to use (0.1.0-small or 0.1.0-base)
            batchsize: Batch size for processing (defaults to 1 for stability)
            no_skipping: Disable failure detection heuristic
            markdown: Enable markdown compatibility
            max_workers: Maximum number of parallel workers (defaults to 2)
            chunk_size: Number of PDFs to process in each batch (defaults to 5)
        """
        self.output_dir = Path(output_dir)
        self.model = model
        self.batchsize = batchsize
        self.no_skipping = no_skipping
        self.markdown = markdown
        self.max_workers = max_workers if max_workers else 2  # Default to 2 for stability
        self.chunk_size = chunk_size  # Small batch size (5) for stability
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.start_time = None
        self.processed_count = 0
        self.total_count = 0
    
    def check_nougat_installation(self) -> bool:
        """
        Check if Nougat is properly installed
        
        Returns:
            True if Nougat is installed, False otherwise
        """
        try:
            result = subprocess.run(['nougat', '--help'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
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
    
    def find_pdf_files(self, directory: str) -> List[Path]:
        """
        Find all PDF files in a directory
        
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
    
    def convert_single_pdf(self, pdf_path: Path) -> bool:
        """
        Convert a single PDF file to text using Nougat
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Check if output already exists and skip if it does
            output_file = self.output_dir / f"{pdf_path.stem}.mmd"
            if output_file.exists():
                logger.debug(f"Skipping {pdf_path.name} - output already exists")
                return True
            
            # Build the command
            cmd = [
                'nougat',
                str(pdf_path),
                '-o', str(self.output_dir),
                '-m', self.model,
                '-b', str(self.batchsize)
            ]
            
            if self.no_skipping:
                cmd.append('--no-skipping')
            
            if self.markdown:
                cmd.append('--markdown')
            else:
                cmd.append('--no-markdown')
            
            logger.debug(f"Converting {pdf_path.name}...")
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.debug(f"Successfully converted {pdf_path.name}")
                return True
            else:
                logger.error(f"Error converting {pdf_path.name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception while converting {pdf_path.name}: {str(e)}")
            return False
        
    def convert_pdf_chunk(self, pdf_paths: List[Path], temp_output_dir: Path) -> Dict[str, bool]:
        """
        Convert a chunk of PDFs to a temporary directory
        
        Args:
            pdf_paths: List of PDF paths to convert
            temp_output_dir: Temporary output directory
            
        Returns:
            Dictionary mapping PDF paths to success status
        """
        # Create a temporary file with list of PDFs
        temp_list_file = temp_output_dir / "pdf_list.txt"
        with open(temp_list_file, 'w') as f:
            for pdf_path in pdf_paths:
                f.write(f"{pdf_path}\n")
        
        # Build the command
        cmd = [
            'nougat',
            str(temp_list_file),
            '-o', str(temp_output_dir),
            '-m', self.model,
            '-b', str(self.batchsize)
        ]
        
        if self.no_skipping:
            cmd.append('--no-skipping')
        
        if self.markdown:
            cmd.append('--markdown')
        else:
            cmd.append('--no-markdown')
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Move results to final output directory
        results = {}
        for pdf_path in pdf_paths:
            output_file = temp_output_dir / f"{pdf_path.stem}.mmd"
            final_output = self.output_dir / f"{pdf_path.stem}.mmd"
            
            if output_file.exists():
                try:
                    # Copy file to final destination
                    with open(output_file, 'r') as src, open(final_output, 'w') as dst:
                        dst.write(src.read())
                    results[str(pdf_path)] = True
                except Exception as e:
                    logger.error(f"Error moving {output_file} to {final_output}: {str(e)}")
                    results[str(pdf_path)] = False
            else:
                results[str(pdf_path)] = False
        
        return results
    
    def process_in_parallel(self, pdf_files: List[Path]) -> Tuple[int, int]:
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
        
        # Create chunks of PDFs
        pdf_chunks = [pdf_files[i:i + self.chunk_size] for i in range(0, len(pdf_files), self.chunk_size)]
        logger.info(f"Processing {len(pdf_files)} PDFs in {len(pdf_chunks)} chunks of up to {self.chunk_size} PDFs each")
        
        # Create temporary directories for each worker
        temp_dirs = []
        for i in range(self.max_workers):
            temp_dir = self.output_dir / f"temp_{i}"
            temp_dir.mkdir(exist_ok=True)
            temp_dirs.append(temp_dir)
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            futures = []
            for i, chunk in enumerate(pdf_chunks):
                temp_dir = temp_dirs[i % self.max_workers]
                futures.append(executor.submit(self.convert_pdf_chunk, chunk, temp_dir))
                
            # Process results as they complete
            for i, future in enumerate(futures):
                try:
                    results = future.result()
                    for pdf_path, success in results.items():
                        if success:
                            successful += 1
                        else:
                            failed += 1
                    
                    self.processed_count += len(pdf_chunks[i])
                    completion_pct = (self.processed_count / self.total_count) * 100
                    est_time = self.estimate_completion_time()
                    
                    logger.info(f"Progress: {completion_pct:.1f}% ({self.processed_count}/{self.total_count}) - Est. time remaining: {est_time}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    failed += len(pdf_chunks[i])
        
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            for file in temp_dir.glob("*"):
                try:
                    file.unlink()
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
        
        return successful, failed
    
    def convert_directory(self, pdf_directory: str, recompute: bool = False) -> None:
        """
        Convert all PDFs in a directory to text
        
        Args:
            pdf_directory: Path to directory containing PDFs
            recompute: Recompute already processed PDFs
        """
        if not self.check_nougat_installation():
            logger.error("Nougat is not installed or not found in PATH")
            logger.info("Install with: pip install nougat-ocr")
            return
        
        # For small number of PDFs or when recompute is needed, use the nougat CLI directly
        if Path(pdf_directory).is_file() or recompute:
            try:
                cmd = [
                    'nougat',
                    str(pdf_directory),
                    '-o', str(self.output_dir),
                    '-m', self.model,
                    '-b', str(self.batchsize)
                ]
                
                if self.no_skipping:
                    cmd.append('--no-skipping')
                
                if self.markdown:
                    cmd.append('--markdown')
                else:
                    cmd.append('--no-markdown')
                    
                if recompute:
                    cmd.append('--recompute')
                
                logger.info(f"Converting PDFs in {pdf_directory}...")
                logger.info(f"Command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Successfully converted PDFs")
                    logger.info(f"Output files saved to: {self.output_dir}")
                else:
                    logger.error(f"Error during conversion: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Exception during conversion: {str(e)}")
            return
        
        # For large directories, use parallel processing
        try:
            pdf_files = self.find_pdf_files(pdf_directory)
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {pdf_directory}")
                return
            
            # Check for very large number of files
            if len(pdf_files) > 2000:
                logger.warning(f"Large number of PDFs detected ({len(pdf_files)}). This may take a long time.")
                logger.info(f"Using {self.max_workers} workers with batch size {self.batchsize}")
                
            # Filter out already processed files unless recompute is True
            if not recompute:
                unprocessed_files = []
                for pdf_file in pdf_files:
                    output_file = self.output_dir / f"{pdf_file.stem}.mmd"
                    if not output_file.exists():
                        unprocessed_files.append(pdf_file)
                
                if len(unprocessed_files) < len(pdf_files):
                    logger.info(f"Skipping {len(pdf_files) - len(unprocessed_files)} already processed files")
                    pdf_files = unprocessed_files
            
            if not pdf_files:
                logger.info("All files have already been processed")
                return
                
            # Process in parallel
            successful, failed = self.process_in_parallel(pdf_files)
            
            # Report results
            logger.info(f"Conversion complete! Success: {successful}, Failed: {failed}")
            logger.info(f"Output files saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Exception during parallel conversion: {str(e)}")
    
    def convert_directory_individually(self, pdf_directory: str) -> None:
        """
        Convert PDFs one by one (useful for progress tracking and error handling)
        
        Args:
            pdf_directory: Path to directory containing PDFs
        """
        if not self.check_nougat_installation():
            logger.error("Nougat is not installed or not found in PATH")
            logger.info("Install with: pip install nougat-ocr")
            return
        
        pdf_files = self.find_pdf_files(pdf_directory)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return
        
        successful = 0
        failed = 0
        total = len(pdf_files)
        
        self.start_time = time.time()
        self.total_count = total
        
        for i, pdf_file in enumerate(pdf_files):
            if self.convert_single_pdf(pdf_file):
                successful += 1
            else:
                failed += 1
            
            self.processed_count = i + 1
            completion_pct = (self.processed_count / total) * 100
            est_time = self.estimate_completion_time()
            
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"Progress: {completion_pct:.1f}% ({i+1}/{total}) - Est. time remaining: {est_time}")
        
        elapsed = time.time() - self.start_time
        logger.info(f"Conversion complete! Success: {successful}, Failed: {failed}")
        logger.info(f"Total time: {timedelta(seconds=int(elapsed))}")
        logger.info(f"Output files saved to: {self.output_dir}")
    
    def read_converted_text(self, pdf_name: str) -> Optional[str]:
        """
        Read the converted text from a .mmd file
        
        Args:
            pdf_name: Name of the original PDF file (without extension)
            
        Returns:
            The converted text content or None if file not found
        """
        mmd_file = self.output_dir / f"{pdf_name}.mmd"
        
        if mmd_file.exists():
            try:
                with open(mmd_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading {mmd_file}: {str(e)}")
                return None
        else:
            logger.warning(f"Converted file not found: {mmd_file}")
            return None


def main():
    """
    Main function to handle command line arguments and run the parser
    """
    parser = argparse.ArgumentParser(description="Convert PDFs to text using Nougat")
    parser.add_argument("pdf_dir", nargs='?', default="test_scrap_mini", 
                       help="Directory containing PDF files (default: test_scrap_mini)")
    parser.add_argument("-o", "--output", default="output_mini", 
                       help="Output directory for converted files (default: output_mini)")
    parser.add_argument("-m", "--model", default="0.1.0-small",
                       choices=["0.1.0-small", "0.1.0-base"],
                       help="Nougat model to use")
    parser.add_argument("-b", "--batchsize", type=int, default=1,
                       help="Batch size for processing (default: 1)")
    parser.add_argument("--no-skipping", action="store_true",
                       help="Disable failure detection heuristic")
    parser.add_argument("--no-markdown", action="store_true",
                       help="Disable markdown compatibility")
    parser.add_argument("--individual", action="store_true",
                       help="Process PDFs individually (for progress tracking)")
    parser.add_argument("--recompute", action="store_true",
                       help="Recompute already processed PDFs")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of worker threads (default: 2)")
    parser.add_argument("--chunk-size", type=int, default=5,
                       help="Number of PDFs to process in each batch (default: 5)")
    parser.add_argument("--recursive", action="store_true",
                       help="Recursively find PDFs in subdirectories")
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
    nougat_parser = NougatPDFParser(
        output_dir=args.output,
        model=args.model,
        batchsize=args.batchsize,
        no_skipping=args.no_skipping,
        markdown=not args.no_markdown,
        max_workers=args.workers,
        chunk_size=args.chunk_size
    )
    
    start_time = time.time()
    
    try:
        # If max-pdfs or start-index is specified, use a special mode
        if args.max_pdfs is not None or args.start_index > 0:
            # Find all PDFs
            all_pdfs = nougat_parser.find_pdf_files(args.pdf_dir)
            
            # Apply start index and max count
            start_idx = args.start_index
            end_idx = len(all_pdfs) if args.max_pdfs is None else min(start_idx + args.max_pdfs, len(all_pdfs))
            
            # Get subset of PDFs
            pdfs_to_process = all_pdfs[start_idx:end_idx]
            logger.info(f"Processing PDF subset: {start_idx} to {end_idx-1} (total: {len(pdfs_to_process)})")
            
            # Process this batch
            if len(pdfs_to_process) > 0:
                if args.individual:
                    # Convert individually
                    successful = 0
                    failed = 0
                    total = len(pdfs_to_process)
                    nougat_parser.start_time = time.time()
                    nougat_parser.total_count = total
                    
                    for i, pdf_file in enumerate(pdfs_to_process):
                        if nougat_parser.convert_single_pdf(pdf_file):
                            successful += 1
                        else:
                            failed += 1
                        
                        nougat_parser.processed_count = i + 1
                        completion_pct = (nougat_parser.processed_count / total) * 100
                        est_time = nougat_parser.estimate_completion_time()
                        
                        if (i + 1) % 2 == 0 or (i + 1) == total:
                            logger.info(f"Progress: {completion_pct:.1f}% ({i+1}/{total}) - Est. time remaining: {est_time}")
                    
                    logger.info(f"Batch conversion complete! Success: {successful}, Failed: {failed}")
                else:
                    # Use parallel processing
                    successful, failed = nougat_parser.process_in_parallel(pdfs_to_process)
                    logger.info(f"Batch conversion complete! Success: {successful}, Failed: {failed}")
            else:
                logger.warning("No PDFs to process in the specified range")
        else:
            # Convert PDFs using standard method
            if args.individual:
                nougat_parser.convert_directory_individually(args.pdf_dir)
            else:
                nougat_parser.convert_directory(args.pdf_dir, recompute=args.recompute)
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
        print("Batch Size: 1")
        print("Workers: 2")
        print("Chunk Size: 5")
        print()
        print("For processing large directories:")
        print("python nougat_parser.py large_pdf_dir -o output_dir --max-pdfs 20 --start-index 0")
        print("python nougat_parser.py large_pdf_dir -o output_dir --max-pdfs 20 --start-index 20")
        print("python nougat_parser.py large_pdf_dir -o output_dir --max-pdfs 20 --start-index 40")
        print()
        print("You can also use custom arguments:")
        print("python nougat_parser.py [pdf_directory] --chunk-size 5 --workers 2")
        print("python nougat_parser.py test_scrap_mini -o output_mini -m 0.1.0-base")
        print("python nougat_parser.py test_scrap_mini --individual --no-skipping")
    
    main()


# Example of how to use this programmatically in another script:
"""
# Import the class
from nougat_parser import NougatPDFParser

# Initialize the parser with small batch settings
parser = NougatPDFParser(
    output_dir="output_mini",
    model="0.1.0-small",  # Small model is faster
    batchsize=1,          # Small batch size for stability
    no_skipping=True,     # Avoid skipping pages
    max_workers=2,        # Limit concurrent workers
    chunk_size=5          # Process in very small batches
)

# Process a small batch of PDFs at a time
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


if __name__ == "__main__":
    # Example usage when running as a script
    if len(sys.argv) == 1:
        print("No arguments provided. Using default settings:")
        print("PDF Directory: test_scrap_mini")
        print("Output Directory: output_mini")
        print()
        print("You can also use custom arguments:")
        print("python nougat_parser.py [pdf_directory]")
        print("python nougat_parser.py test_scrap_mini -o output_mini -m 0.1.0-base")
        print("python nougat_parser.py test_scrap_mini --individual --no-skipping")
    
    main()


# Example of how to use this programmatically in another script:
"""
# Import the class
from nougat_parser import NougatPDFParser

# Initialize the parser with default directories
parser = NougatPDFParser(
    output_dir="output_mini",
    model="0.1.0-base",
    batchsize=2,
    no_skipping=True
)

# Convert PDFs from test_scrap_mini directory
parser.convert_directory("test_scrap_mini")

# Or convert PDFs individually for better progress tracking
parser.convert_directory_individually("test_scrap_mini")

# Read the converted text
text_content = parser.read_converted_text("example_paper")
if text_content:
    print(text_content)
"""
import os
import subprocess
import time
import glob
from pathlib import Path

# Configuration
PDF_DIR = "test_scrap"  # Replace with your PDF directory
OUTPUT_DIR = "output_parser_pdf"   # Replace with your output directory
BATCH_SIZE = 20             # Number of PDFs to process in each batch

# Optional parameters
MODEL = "0.1.0-small"       # Model to use (0.1.0-small or 0.1.0-base)
WORKERS = 2                 # Number of worker threads
CHUNK_SIZE = 5              # PDFs per chunk

# Create log and output directories
os.makedirs("logs", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting batch processing of PDFs from {PDF_DIR}")
print(f"Output will be saved to {OUTPUT_DIR}")
print(f"Processing in batches of {BATCH_SIZE} PDFs")

# Count PDFs to process
print("Scanning for PDF files...")
pdf_files = list(Path(PDF_DIR).glob("**/*.pdf"))
pdf_files.extend(list(Path(PDF_DIR).glob("**/*.PDF")))
total_files = len(pdf_files)

if total_files == 0:
    print(f"No PDF files found in {PDF_DIR}. Please check the directory path.")
    exit(1)

print(f"Found {total_files} PDF files to process")

# Calculate number of batches
num_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
print(f"Will process in {num_batches} batches")

# Process in batches
for i in range(0, total_files, BATCH_SIZE):
    batch_num = i // BATCH_SIZE + 1
    end_idx = min(i + BATCH_SIZE, total_files)
    batch_size = end_idx - i
    log_file = f"logs/batch_{batch_num}.log"
    
    print(f"\nProcessing batch {batch_num}/{num_batches} (PDFs {i} to {end_idx-1}, {batch_size} files)")
    print(f"Logging to {log_file}")
    
    # Build the command
    cmd = [
        "python", "nougat_pdf_parser.py", 
        PDF_DIR,
        "-o", OUTPUT_DIR,
        "-m", MODEL,
        "--workers", str(WORKERS),
        "--chunk-size", str(CHUNK_SIZE),
        "--max-pdfs", str(batch_size),
        "--start-index", str(i),
        "--no-skipping"
    ]
    
    # Run the command
    with open(log_file, "w") as log:
        start_time = time.time()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        
        print("Processing batch... ", end="")
        dots = 0
        while proc.poll() is None:
            print(".", end="", flush=True)
            dots += 1
            if dots % 30 == 0:  # Start a new line every 30 dots for readability
                print("")
                print("Still processing... ", end="")
            time.sleep(5)
        
        elapsed = time.time() - start_time
        minutes, seconds = divmod(elapsed, 60)
        print(f" Done! (Took {int(minutes)}m {int(seconds)}s)")
    
    # Check progress
    processed_count = len(glob.glob(os.path.join(OUTPUT_DIR, "*.mmd")))
    print(f"Processed {processed_count}/{total_files} PDFs so far ({processed_count/total_files*100:.1f}%)")
    
    if batch_num < num_batches:
        # Optional: add a short pause between batches
        print("Pausing for 5 seconds before next batch...")
        time.sleep(5)

print("\nAll batches completed!")
final_count = len(glob.glob(os.path.join(OUTPUT_DIR, "*.mmd")))
print(f"Total PDFs processed: {final_count}/{total_files} ({final_count/total_files*100:.1f}%)")

# Check for any PDFs that didn't get processed
if final_count < total_files:
    print(f"Warning: {total_files - final_count} PDFs were not processed successfully.")
    print("Check the log files in the 'logs' directory for details.")

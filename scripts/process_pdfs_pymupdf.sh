#!/bin/bash
# Script to process a large number of PDFs in small batches using PyMuPDF
# Make this file executable with: chmod +x process_pdfs.sh

# Configuration
PDF_DIR="test_scrap_mini"  # Replace with your PDF directory
OUTPUT_DIR="output_mini"   # Replace with your output directory
BATCH_SIZE=20              # Number of PDFs to process in each batch
TOTAL_PDFS=2000            # Approximate total number of PDFs (can be larger)

# Optional parameters
MODE="text"                # text, html, blocks, words, json, etc.
WORKERS=2                  # Number of worker threads
CHUNK_SIZE=5               # PDFs per chunk

# Create log directory
mkdir -p logs

echo "Starting batch processing of PDFs from $PDF_DIR"
echo "Output will be saved to $OUTPUT_DIR"
echo "Processing in batches of $BATCH_SIZE PDFs using PyMuPDF"

# Process in batches
for ((i=0; i<TOTAL_PDFS; i+=BATCH_SIZE)); do
  BATCH_NUM=$((i / BATCH_SIZE + 1))
  LOG_FILE="logs/batch_${BATCH_NUM}.log"
  
  echo "Processing batch $BATCH_NUM (PDFs $i to $((i+BATCH_SIZE-1)))"
  echo "Logging to $LOG_FILE"
  
  # Run the Python script for this batch
  python pymupdf_parser.py "$PDF_DIR" \
    -o "$OUTPUT_DIR" \
    -m "$MODE" \
    --workers "$WORKERS" \
    --chunk-size "$CHUNK_SIZE" \
    --max-pdfs "$BATCH_SIZE" \
    --start-index "$i" \
    2>&1 | tee "$LOG_FILE"
  
  # Check if we need to continue
  PROCESSED_COUNT=$(find "$OUTPUT_DIR" -name "*.txt" | wc -l)
  echo "Processed $PROCESSED_COUNT PDFs so far"
  
  # Optional: add a short pause between batches
  echo "Pausing for 5 seconds before next batch..."
  sleep 5
done

echo "All batches completed!"
echo "Total PDFs processed: $(find "$OUTPUT_DIR" -name "*.txt" | wc -l)"
#!/bin/bash
# Auto pipeline: wait for the running `cp` (copying Apr11_relaxed_all_archives
# from Google Drive into ~/data/trajectory_apr11) to finish, then automatically
# extract every .tar.zst archive with zstd, and log everything.
#
# Usage:
#   chmod +x ~/Diff-SCM/auto_pipeline.sh
#   tmux new -s autopipeline
#   ~/Diff-SCM/auto_pipeline.sh
#   (Ctrl+B then D to detach; reattach anytime with: tmux attach -t autopipeline)
#
# Progress / results land in ~/auto_pipeline.log

set -u

LOG=~/auto_pipeline.log
DATA_DIR=/mnt/h/trajectory_apr11/Apr11_relaxed_all_archives

echo "=== $(date) :: auto_pipeline started ===" > "$LOG"

# --- Step 1: wait until the copy process is done -----------------------------
echo "[$(date)] Waiting for 'cp' (Apr11_relaxed_all_archives) to finish..." >> "$LOG"

while pgrep -af "cp -rv" 2>/dev/null | grep -q "Apr11_relaxed_all_archives"; do
    SIZE=$(du -sh ~/data/trajectory_apr11/ 2>/dev/null | cut -f1)
    echo "[$(date)] still copying... current size: ${SIZE:-unknown}" >> "$LOG"
    sleep 120
done

echo "[$(date)] Copy process is no longer running. Proceeding to extraction." >> "$LOG"

# --- Step 2: sanity check the data directory ---------------------------------
if [ ! -d "$DATA_DIR" ]; then
    echo "[$(date)] ERROR: $DATA_DIR does not exist. Aborting." >> "$LOG"
    exit 1
fi

cd "$DATA_DIR" || { echo "[$(date)] ERROR: cannot cd into $DATA_DIR" >> "$LOG"; exit 1; }

echo "[$(date)] Contents of $DATA_DIR:" >> "$LOG"
ls -la "$DATA_DIR" >> "$LOG" 2>&1

# --- Step 3: make sure zstd is available --------------------------------------
if ! command -v unzstd >/dev/null 2>&1; then
    echo "[$(date)] unzstd not found, installing zstd (requires sudo password)..." >> "$LOG"
    sudo apt-get update >> "$LOG" 2>&1
    sudo apt-get install -y zstd >> "$LOG" 2>&1
fi

# --- Step 4: extract every .tar.zst archive ----------------------------------
shopt -s nullglob
ARCHIVES=( *.tar.zst )

if [ ${#ARCHIVES[@]} -eq 0 ]; then
    echo "[$(date)] No .tar.zst files found in $DATA_DIR. Nothing to extract." >> "$LOG"
else
    for f in "${ARCHIVES[@]}"; do
        base="${f%.tar.zst}"
        base="${base#Copy of }"
        outdir="extracted_${base}"
        if [ -f "$outdir/.extraction_complete" ]; then echo "[$(date)]   Skipping '$f' -> '$outdir' already fully extracted (marker found), skipping." >> "$LOG"; continue; fi; mkdir -p "$outdir"
        echo "[$(date)] Extracting '$f' -> '$outdir' ..." >> "$LOG"
        if tar --use-compress-program=unzstd -xf "$f" -C "$outdir" >> "$LOG" 2>&1; then
            echo "[$(date)]   OK: $f extracted." >> "$LOG"; touch "$outdir/.extraction_complete"
        else
            echo "[$(date)]   FAILED: $f (see log above for tar error)" >> "$LOG"
        fi
    done
fi

# --- Step 5: report what data files were produced ----------------------------
echo "[$(date)] Searching for .h5 / .hdf5 files under $DATA_DIR ..." >> "$LOG"
find "$DATA_DIR" -iname "*.h5" -o -iname "*.hdf5" >> "$LOG" 2>&1

echo "[$(date)] Disk usage of extracted data:" >> "$LOG"
du -sh "$DATA_DIR"/extracted_* >> "$LOG" 2>&1

echo "=== $(date) :: auto_pipeline FINISHED ===" >> "$LOG"
echo "" >> "$LOG"
echo "Next step: point the training script at one of the 'extracted_*' folders, e.g.:" >> "$LOG"
echo "  python diff_scm/training/trajectory_step_diffusion_train.py --data-path \"$DATA_DIR/extracted_<name>/\"" >> "$LOG"

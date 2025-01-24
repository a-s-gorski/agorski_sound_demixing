!/bin/bash

URL to download
URL="https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"

# Output filename
OUTPUT_FILE="musdb18hq.zip"

DEST_DIR="data/musdb18_hq"

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    echo "Downloading using wget..."
    wget -O "$OUTPUT_FILE" "$URL"
elif command -v curl &> /dev/null; then
    echo "Downloading using curl..."
    curl -L "$URL" -o "$OUTPUT_FILE"
else
    echo "Error: Neither wget nor curl is installed. Please install one and try again."
    exit 1
fi

echo "Download completed: $OUTPUT_FILE"

mkdir -p "$DEST_DIR"
unzip musdb18hq.zip -d "$DEST_DIR"

if [ $? -eq 0 ]; then
    echo "Unzipping successful. Deleting zip file..."
    rm -f "$ZIP_FILE"
else
    echo "Unzipping failed. Zip file not deleted."
fi
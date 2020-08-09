#!/sh
if [ ! -z "$1" ]; then
    python -m decompress.task --zip_file nsynth-$1.jsonwav.tar.gz
fi
#!/usr/bin/env sh

for file in $(ls images); do
  filename="${file%.*}".png
  echo "$file -> $filename"
  python extract.py images/$file assets/$filename --loop
done

#!/bin/bash

# Set the path to your files
files_path="/path/to/your/files"

# Set the number of files per subfolder
files_per_subfolder=10000

# Create a subfolder for each batch of files
find "$files_path" -maxdepth 1 -type f | awk -v files_per_subfolder="$files_per_subfolder" '{
  i = int((NR-1)/files_per_subfolder) + 1
  subfolder = sprintf("Subfolder_%04d", i)
  print "mkdir -p \"" files_path "/" subfolder "\"; mv \"" $0 "\" \"" files_path "/" subfolder "\""
}' | sh
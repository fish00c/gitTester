#!user/bin/bash

unzip $1/unsplit.zip -d $2

echo "Today is " `date`

echo "the current path is" `pwd`
echo "the data file path is $1"
echo "the destination path is $2" 


echo " your data file path has the following files and folders: "
ls $1

echo " your destination path has the following files and folders: "
ls $2


selected_subfolder=""

for subfolder in "$2"/*/; do
	echo $subfolder
    if [[ "$subfolder" == *imagenet* ]]; then
        selected_subfolder="$subfolder"
        break
    fi
done

# Check if a matching subfolder was found
if [ -z "$selected_subfolder" ]; then
    echo "No subfolder containing the term 'imagenet' found in '$2'."
    exit 1
fi

echo " number of images in the Train AI folder"
ls $selected_subfolder/train/ai | wc -l

echo " number of images in the Train Nature folder"
ls $selected_subfolder/train/nature | wc -l

echo " number of images in the Val AI folder"
ls $selected_subfolder/val/ai | wc -l

echo " number of images in the Val Narure folder"
ls $selected_subfolder/val/nature | wc -l
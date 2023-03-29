# Installation
cd pymeanshift
./setup.py install
pip install imutils

# Run
python main.py --level=1
python main.py --level=1 --nums_of_spots=4
python3 find_the_differences.py --img1 ./input/input_image.png --img2  ./output/output.png

python3 find_the_differences.py --img1 input/input_lv3.png --img2  level_3/output.png
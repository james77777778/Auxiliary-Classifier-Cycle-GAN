# Download dataset
mkdir -p dataset
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip -P datast
unzip dataset/summer2winter_yosemite -d dataset/
mkdir dataset/summer2winter_yosemite/train
mkdir dataset/summer2winter_yosemite/test
# Change dataset structure
mv dataset/summer2winter_yosemite/testA dataset/summer2winter_yosemite/test/
mv dataset/summer2winter_yosemite/testB dataset/summer2winter_yosemite/test/
mv dataset/summer2winter_yosemite/trainA dataset/summer2winter_yosemite/train/
mv dataset/summer2winter_yosemite/trainB dataset/summer2winter_yosemite/train/
mv dataset/summer2winter_yosemite/test/testA dataset/summer2winter_yosemite/test/A
mv dataset/summer2winter_yosemite/test/testB dataset/summer2winter_yosemite/test/B
mv dataset/summer2winter_yosemite/train/trainA dataset/summer2winter_yosemite/train/A
mv dataset/summer2winter_yosemite/train/trainB dataset/summer2winter_yosemite/train/B
# Train
python3 train.py --flagfile config/summer2winter_yosemite.cfg
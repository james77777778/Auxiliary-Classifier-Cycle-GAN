# Download dataset
mkdir -p dataset
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip -P dataset
unzip dataset/facades.zip -d dataset/
mkdir dataset/facades/train
mkdir dataset/facades/test
# Change dataset structure
mv dataset/facades/testA dataset/facades/test/
mv dataset/facades/testB dataset/facades/test/
mv dataset/facades/trainA dataset/facades/train/
mv dataset/facades/trainB dataset/facades/train/
mv dataset/facades/test/testA dataset/facades/test/A
mv dataset/facades/test/testB dataset/facades/test/B
mv dataset/facades/train/trainA dataset/facades/train/A
mv dataset/facades/train/trainB dataset/facades/train/B
# Train
python3 train.py --flagfile config/facades.cfg
# GAN-FID

This project will research the effect of fid(Fréchet Inception Distance) on training
a GAN/DCGAN network.

train.py - the main training file of the DCGAN. can train with different methods that include FID.
fid.py - script to calculate the FID score between 2 folders with samples in it.
train_fid.py - script to train over pretrained model with FID score only.
create_dataset_images.py - help to get real samples of a dataset.
generate_dataset_images.py - generates images from the trained generator to compare after.
calc_stat.py - script to improve times of training, we can calculate all the values for the fid calculation beforehand.
dcgan.py - my implementation of DCGAN.
fid_utils.py - all fastFID functions
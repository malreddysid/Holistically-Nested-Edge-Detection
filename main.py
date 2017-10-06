from data_loader import DataLoader
from hed import HED

path = '/Users/siddarth/Desktop/apple/HED-BSDS/'
train_list_file = '/Users/siddarth/Desktop/apple/HED-BSDS/train_pair.lst'
checkpoint_dir = '/Users/siddarth/Desktop/apple/chckpnt_dir/'
weights_path = '/Users/siddarth/Desktop/apple/VGG_weights.npy'

data_loader = DataLoader(data_path=path, file_list=train_list_file)

num_epochs = 100 # Number of epochs to train
summary_write_freq = 100
model_save_freq = 10000

hed = HED(data_loader=data_loader, num_epochs=num_epochs, chckpnt_dir=checkpoint_dir, weights_path=weights_path, summary_write_freq=summary_write_freq, model_save_freq=model_save_freq)
hed.train()

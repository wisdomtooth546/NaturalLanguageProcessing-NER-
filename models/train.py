from model import nerModel
from  dataprocessor import dataPreprocessor
import argparse, os, sys


parser = argparse.ArgumentParser(description='Input the files required to train the model. The file path, epochs (default = 2), maximum length of the sequence (default = 75)')
parser.add_argument('path',  type=str, default='data/train.tsv', help='used to pass the path to the training data')
parser.add_argument('-e','--epochs',default = 1, help="takes the number of training cycles")
parser.add_argument('-p', '--model_path', type=str, default = './models/TrainedModels/', help= "Location to save the model")
parser.add_argument('-m', '--mode', type=str, default = 'train', help = "used to put the model in training ('train') and inference mode ('infer')")


cli_inputs = parser.parse_args()
epochs = cli_inputs.epochs
max_len = 75
batch_size = 32
train_path = cli_inputs.path
mode = cli_inputs.mode                     # Put model into train mode
model_path = cli_inputs.model_path

if not os.path.isfile(train_path):
    print(f'The path specified does not exist\n {train_path}')
    sys.exit()

 
           
processor = dataPreprocessor(batch_size, max_len)                                   # An object of dataPtrprocessor
processor.setPath(train_path)
train_dataloader, valid_dataloader, Tags = processor.generate_dataloaders()         # Generating data loaders

model = nerModel(mode, max_len, batch_size)                                         # Initialize the model
model.setTrainingParams(epochs, model_path)
model.provide_data(train_dataloader, valid_dataloader, Tags)                        # Load the data loaders into the model
model.train_model()                                                                 # Train the model

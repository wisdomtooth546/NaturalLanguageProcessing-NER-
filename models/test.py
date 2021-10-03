from model import nerModel
from  dataprocessor import dataPreprocessor
import argparse, os, sys
from seqeval.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser(description='Input the files required to test the model. The file path, epochs (default = 2), maximum length of the sequence (default = 75)')
parser.add_argument('path', type=str, default='data/train.tsv', help='used to pass the path to test the model')
parser.add_argument('-p', '--model_path', type=str, default = './models/TrainedModels/')

cli_inputs = parser.parse_args()
test_path = cli_inputs.path
model_path = cli_inputs.model_path
max_len = 75
batch_size = 32
Tags = ['B', 'O', 'I', 'PAD']

if not os.path.isfile(test_path):
    print(f'The test data path specified does not exist\n {test_path}')
    sys.exit()

if not os.listdir(model_path):
    print(f'No model found in {model_path}')
    sys.exit()

processor = dataPreprocessor(batch_size, max_len)             # Initialize the data processor

processor.setPath(test_path)                                  # Set path

model = nerModel('eval', max_len, batch_size)                 # Initialize the model

model.loadSavedModel(model_path)                           # Load the trained model

model.evalMode()                                              # Put model into inference mode



testSentences, testLabels = processor.testDatatoSentences()
tokenCollection = []
labelCollection = []
test_loss = 0

for testSentence in testSentences:
    tokenized_sentence = processor.tokenizeSentences(testSentence)
    prediction, loss = model.predict(tokenized_sentence)
    test_loss += loss  
            # join bpe split tokens
    tokens = processor.idsToTokens(tokenized_sentence)
    
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, prediction[0]):
            if token == '[CLS]' or token == '[SEP]':
                continue
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(Tags[label_idx])
                new_tokens.append(token)
    tokenCollection.append(new_tokens)
    labelCollection.append(new_labels)

print("Test Loss: {}".format(test_loss/len(testSentences)))
print("Test Accuracy: {}".format(accuracy_score(testLabels, labelCollection)))
print("Classification Report:\n {}".format(classification_report(testLabels, labelCollection)))

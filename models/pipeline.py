from model import nerModel
from  dataprocessor import dataPreprocessor
import argparse, os, sys


parser = argparse.ArgumentParser(description='Input the sentences to be tagged')
parser.add_argument('sentence',  type=str, help='used to pass the sentence')
parser.add_argument('-p', '--model_path', type=str, metavar = 'model_path', default = './models/TrainedModels/SavedModel/SubmissionModel/')


cli_inputs = parser.parse_args()
testSentence = cli_inputs.sentence
model_path = cli_inputs.model_path
batch_size = 32
max_len = 75
print(testSentence)

processor = dataPreprocessor(batch_size, max_len)             # Initialize the data processor
Tags = ['B', 'O', 'I', 'PAD']

if not os.listdir(model_path):
    print(f'No model found in {model_path}')
    sys.exit()

model = nerModel('eval', max_len, batch_size)                 # Initialize the model

model.loadSavedModel(model_path)                              # Load the trained model

model.evalMode()                                              # Put model into inference mode

tokenized_sentence = processor.tokenizeSentences(testSentence)
prediction, loss = model.predict(tokenized_sentence)

# join bpe split tokens
tokens = processor.idsToTokens(tokenized_sentence)
    
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, prediction[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(Tags[label_idx])
        new_tokens.append(token)

for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np


class dataPreprocessor():


        def __init__(self, batch_size, max_len):
            self.batch_size = batch_size
            self.max_len = max_len
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)   # BERT Tokenizer based on WordPiece vocab

        def setPath(self, path):
            self.path = path
                    
        def dataToLabels(self, filename):             # Function to extract words and tags from .tsv file
            datafile = open(filename)
            datanLabels = []
            sentence = []
            tags = []
            for line in datafile:
                if len(line)==0 or line[0]=="\n":
                    if len(sentence) > 0:
                        datanLabels.append(sentence)
                        sentence = []
                        continue
                splits = line.split('\t')
                sentence.append([splits[0],splits[-1].rstrip("\n")])
        
            if len(sentence) > 0:
                datanLabels.append(sentence)
                sentence = []
            return datanLabels

        def extractWordsNTags(self, data):        # Function to split words and Tags
            sentences = [[word[0] for word in sentence] for sentence in data]
            labels = [[word[1] for word in sentence] for sentence in data]
            return sentences, labels

        
        
        def tokenize_and_preserve_labels(self, sentence, text_labels):  # Funtion to tokenize the words into subwords based on WordPiece vocab from BERT
            tokenized_sentence = []
            labels = []

            for word, label in zip(sentence, text_labels):
                tokenized_word = self.tokenizer.tokenize(word)            # Tokenize the word and count # of subwords the word is broken into
                n_subwords = len(tokenized_word)
                tokenized_sentence.extend(tokenized_word)             # Add the tokenized word to the final tokenized word list
                labels.extend([label] * n_subwords)                 # Add the same label to the new list of labels `n_subwords` times
        
            return tokenized_sentence, labels


        def generate_dataloaders(self):

            data = self.dataToLabels(self.path)
            sentences, labels = self.extractWordsNTags(data)
            self.Tags = list(set(labels[0]))
            self.Tags.append('PAD')
            self.tag2idx = {t: i for i, t in enumerate(self.Tags)}

            #self.tokenizer = AutoTokenizer.from_pretrained("fidukm34/biobert_v1.1_pubmed-finetuned-ner-finetuned-ner")  #BioBERT Tokenizer
        
            tokenized_texts_and_labels = [
                                 self.tokenize_and_preserve_labels(sent, labs)
                                    for sent, labs in zip(sentences, labels)
                                    ]
            tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
            labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

            input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],  # Pad the tokens to be of length = max_len
                          maxlen = self.max_len, dtype="long", value=0.0,
                          truncating="post", padding="post")

            tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],                            # Pad the Tags to be of length = max_len
                     maxlen = self.max_len, value = self.tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

            attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

            train_inputs, valid_inputs, train_tags, valid_tags = train_test_split(input_ids, tags,
                                                            random_state=2021, test_size=0.1)
        
            train_masks, valid_masks, _, _ = train_test_split (attention_masks, input_ids,
                                             random_state=2021, test_size=0.1)
                                             
            train_inputs = torch.tensor(train_inputs)
            valid_inputs = torch.tensor(valid_inputs)
            train_tags = torch.tensor(train_tags)
            valid_tags = torch.tensor(valid_tags)
            train_masks = torch.tensor(train_masks)
            valid_masks = torch.tensor(valid_masks)
        
            train_data = TensorDataset(train_inputs, train_masks, train_tags)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

            valid_data = TensorDataset(valid_inputs, valid_masks, valid_tags)
            valid_sampler = SequentialSampler(valid_data)
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size)

            return train_dataloader, valid_dataloader, self.Tags
        
        def testDatatoSentences(self):
                dataPath = self.path
                fileContents = open(dataPath)
                sentenceCollection = []
                sentenceTags = []
                tagCollection = []
                flag = 1
                for line in fileContents:
                    if flag == 1:
                        firstSplit = line.split('\t')
                        Word = firstSplit[0]
                        tag = firstSplit[-1].rstrip('\n')
                        sentenceTags.append(tag)
                        flag = 0
                        continue
                    if line == '\n':
                        sentenceCollection.append(Word)
                        tagCollection.append(sentenceTags)
                        sentenceTags = []
                        flag = 1
                        continue
                    splitted = line.split('\t')
                    subsqWord = splitted[0]
                    tag = splitted[-1].rstrip('\n')
                    Word = Word + " " + subsqWord
                    sentenceTags.append(tag)
                    
                return sentenceCollection, tagCollection

        
        def tokenizeSentences(self, sentence):
            tokenized_sentence = self.tokenizer.encode(sentence)
            return tokenized_sentence

        def idsToTokens(self, ids):
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            return tokens

        
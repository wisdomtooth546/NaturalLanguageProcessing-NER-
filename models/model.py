import numpy as np
from tqdm import trange
from transformers import BertForTokenClassification, AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch, os
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import accuracy_score, classification_report
import pandas as pd


class nerModel():

    def __init__(self, mode, max_len, batch_size, FULL_FINETUNING = True):
        ## Initialize the model parameters
        self.max_len = max_len
        self.batch_size = batch_size
        self.max_grad_norm = 1.0
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def setTrainingParams(self, epochs, model_path):
        self.epochs = epochs
        self.model_path = model_path
        self.model = AutoModelForTokenClassification.from_pretrained(
                          'bert-base-cased',                        # Importing the pretrained BERT CASED tokenizer
                          num_labels = 4,                           #{'B', 'I', 'O', 'PAD'}
                          output_attentions = False,
                          output_hidden_states = False
                        )


    def provide_data(self, train_dataloader, valid_dataloader, Tags):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.Tags = Tags
        FULL_FINETUNING = True

        if FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.0}
                                ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        self.optimizer = AdamW(
                        optimizer_grouped_parameters,
                        lr = 3e-5,
                        eps = 1e-8
                    )


        self.scheduler = get_linear_schedule_with_warmup(
                                        self.optimizer,
                            num_warmup_steps = 0,
                            num_training_steps = len(self.train_dataloader) * self.epochs
                            )


    def train_model(self):
        
        self.model.to(self.device)
        train_dataloader = self.train_dataloader
        epochs = self.epochs
        optimizer = self.optimizer
        scheduler = self.scheduler
        max_grad_norm  = self.max_grad_norm
        loss_values, validation_loss_values = [], []  ## Store the average loss after each epoch 
        if self.mode == 'train':                            # Train the model
            
            for epoch in trange(epochs, desc="Epoch"):      # ========================================
                                                            #               Training                 
                                                            # ========================================  Perform one full pass over the training set.
                self.model.train()                                      # Put the model into training mode.            
                total_loss = 0                                     # Reset the total loss for this epoch.
                for step, batch in enumerate(train_dataloader):    # Training loop
                    
                    batch = tuple(t.to(self.device) for t in batch)     # add batch to gpu
                    batch_input_ids, batch_input_mask, batch_labels = batch
                    
                    self.model.zero_grad()
                    outputs = self.model(batch_input_ids, 
                                    token_type_ids=None,
                                    attention_mask=batch_input_mask, 
                                    labels=batch_labels)
                    
                    loss = outputs[0]                               # get the loss
                    
                    loss.backward()                                 # Perform a backward pass to calculate the gradients.
                    
                    total_loss += loss.item()                       # track train loss
                    
                    torch.nn.utils.clip_grad_norm_(parameters= self.model.parameters(), max_norm = max_grad_norm) # Clip the norm of the gradient to prevent gradient explosion
                    
                    optimizer.step()
                    
                    scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)     # Calculate the average loss over the training data.
            print("Average train loss: {}".format(avg_train_loss))
            loss_values.append(avg_train_loss)                      # Store the loss value 


            # Put the model into evaluation mode for validation
            self.model.eval()                                            # ========================================
            # Reset the validation loss for this epoch.                  #               Validation
            eval_loss = 0                                                # ========================================
            predictions , true_labels = [], []
            for batch in self.valid_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                batch_input_ids, batch_input_mask, batch_labels = batch

        
                with torch.no_grad():
                    outputs = self.model(batch_input_ids, 
                                token_type_ids = None,              # Forward pass to calculate logit predictions.
                                attention_mask = batch_input_mask, 
                                labels=batch_labels)
        
                # Move logits and labels to CPU
                logits = outputs[1].detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()

                                                                    # Calculate the accuracy for this batch of test sentences.
                eval_loss += outputs[0].mean().item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)

            eval_loss = eval_loss / len(self.valid_dataloader)
            validation_loss_values.append(eval_loss)
            
            print("Validation loss: {}".format(eval_loss))

            pred_tags = [self.Tags[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if self.Tags[l_i] != "PAD"]
            
            valid_tags = [self.Tags[l_i] for l in true_labels
                                  for l_i in l if self.Tags[l_i] != "PAD"]
            
            print("Validation Accuracy: {}\n".format(accuracy_score(pred_tags, valid_tags)))

            self.model.save_pretrained(self.model_path)
        
        print("Model trained successfully and stored at {}".format(self.model_path))
        print(f"The order of Tags used to train the model is {self.Tags}")

    
    def sentenceProcessor(self, sentence):
            tokenized_sentence = self.tokenizer.encode(sentence)
            input_ids = torch.tensor([tokenized_sentence]).cuda()
            with torch.no_grad():
                    outputs = self.model(input_ids)
            label_indices = np.argmax(outputs[0].to('cpu').numpy(), axis=2)
            return label_indices
    
    def loadSavedModel(self, model_path):
        self.model = AutoModelForTokenClassification.from_pretrained(
                            model_path,
                             num_labels = 4,                            ## {'B', 'I', 'O', 'PAD'}
                             output_attentions = False,
                             output_hidden_states = False
                        )
        

    def evalMode(self):    
        self.model.eval()

    def predict(self, tokenized_sentence):
        input = torch.tensor([tokenized_sentence]).to(self.device)
        with torch.no_grad():
            outputs = self.model(input)
        label_indices = np.argmax(outputs[0].to('cpu').numpy(), axis=2)
        loss = outputs[0].mean().item()
        return label_indices, loss
  

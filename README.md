OSCER NER
==============================

This is a Named Entitition project based on Google Al's BERT.
The aim of the project was to fine-tune BERT to clinical data for NER using the data provided. 
This model was able to achieve an accuracy of 97% and an F1 score of 76% on the test dataset.


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── data               <- The data used for this project.
    |
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    |     |
    |     ├── /model.py      <- Main class of the model.
    |     | 
    |     ├── /train.py      <- Script of train the model. 
    |     |     |
    |     |     └ Input: the path to the training data.
    |     |     └ Output: The trained model -> files/trained_model.pt
    |     | 
    |     ├── /test.py       <- Script to run the model on test set.
    |     |      |
    |     |      └ Input: the path to the test data and provides the test accuracy and the classification report.
    |     |      └ Output: files/test_results.tsv
    |     |
    |     ├── /pipeline.py   <- A pipeline to use the trained model to detect entities (NER).
    |     |       |
    |     |       └ Input: str: a sentence
    |     |       └ Output: List: a list of entities B, I, O.
    |     ├── /TrainedModels   <- A folder to hold all the trained models.
    |     |       |
    |     |       └ SubmissionModel: <- MY PROJECT SUBMISSION MODEL.
    |     |       
    |     ├── Files   <- Location to hold the script output files.
    |           └ SubmissionResults.tsv:    <- MY PROJECT SUBMISSION RESULT.
    |
    ├── notebooks          <- Jupyter notebook. 
    │
    ├── references         <- Reference materials used for this project.
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.

    1) train.py

        usage: train.py [-h] [-e EPOCHS] [-p MODEL_PATH] [-m MODE] path

        input the files required to TRAIN the model. 

        inputs: the file path, epochs (default = 2), maximum length of the sequence (default = 75)

        positional arguments:
            path                  used to pass the path to the training data

        optional arguments:
            -h, --help            show this help message and exit
            -e, --epochs          takes the number of training cycles
            -p, --model_path      Location to save the model
            -m, --mode            used to put the model in training ('train') and inference mode ('infer')

    2) test.py
    
        usage: test.py [-h] [-p model_path] path

        input the files required to TEST the model. 

        inputs: the file path, epochs (default =2), maximum length of the sequence (default = 75)

        positional arguments:
           path                  used to pass the path to test the model

        optional arguments:
          -h, --help            show this help message and exit
          -p, --model_path      path to save the model after training
          -t, --tags            order of tags used while training the model. Default is ['B', 'O', 'I', 'PAD']

    3) pipeline.py

         usage: pipeline.py [-h] [-p model_path] sentence

         input the sentences to be tagged

         positional arguments:
            sentence              used to pass the sentence    
                                  eg: "Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia."

         optional arguments:
             -h, --help            show this help message and exit
             -p, --model_path     path to the saved model
             -t, --tags            order of tags used while training the model. Default is ['B', 'O', 'I', 'PAD']

--------

  

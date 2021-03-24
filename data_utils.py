import torch
import pandas as pd 

def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.

     NOTE: This method should featurize the entire dataset simultaneously,
     rather than row-by-row.

  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.

  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """
    ## TODO: Tokenize the questions and passages using both truncation and padding.
    ## Use the tokenizer provided in the argument and see the code comments above for
    ## more details.

    # You can assume that the tokenizer is an instance of a Hugging Face PreTrainedTokenizerFast
    #object that you can simply call on the input dataset argument

    tokenized = tokenizer(dataset["question"].to_list(),dataset["passage"].to_list(),truncation=True,max_length=max_seq_length,padding="max_length", return_tensors="pt")
    
    return tokenized["input_ids"], tokenized["attention_mask"] 


def extract_labels(dataset):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    ## TODO: Convert the labels to a numeric format and return as a list.

    labels = dataset['label'].to_list() #convert the pandas column to a python list

    #convert each true entry to 1 (int) and false entry to 0 (int)

    for n, i in enumerate(labels):
        if i == True:
            labels[n] = 1
        elif i == False:
            labels[n] = 0
            
    return labels




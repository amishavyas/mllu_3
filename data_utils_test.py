import data_utils
import pandas as pd
import torch
import unittest
from collections import Counter

from transformers import RobertaTokenizerFast


class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        self.assertEqual(self.max_seq_len, 4)

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.

        #input_ids.shape == (len(self.dataset) , self.max_seq_len)
        #input_ids.dtype == torch.long

        input_ids, attention_mask = data_utils.encode_data(self.dataset, self.tokenizer, self.max_seq_len) #getting the output of encode_data()
        shape = (len(self.dataset), self.max_seq_len)
        tensor_type = torch.long

        self.assertEqual(input_ids.shape, shape) #comparing the shape of input_ids
        self.assertEqual(attention_mask.shape, shape) #comparing the shape of attention_masks
        
        self.assertEqual(input_ids.dtype, tensor_type) #comparing the type of input_ids
        self.assertEqual(attention_mask.dtype, tensor_type) #comparing the type of attention_masks

        

    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        
        labels = data_utils.extract_labels(self.dataset) #getting the output labels list of extract_labels()

        for i in range(len(labels)):
            
            if self.dataset['label'][i]: #if True
                self.assertEqual(1, labels[i]) #corresponding value in labels list should be equal to 1 
                
            else: #if False 
                self.assertEqual(0, labels[i]) #corresponding value in labels list should be equal to 0 
                
        

if __name__ == "__main__":
    unittest.main()

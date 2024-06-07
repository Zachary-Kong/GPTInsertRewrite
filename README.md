# GPTInsertRewrite

GPT-2 LM was enabled to perform insertion and rewriting tasks through finetuning and prompt engineering.

Specifically, given a predetermined prefix and suffix, the model was to perform insert/rewrite of text between the prefix and suffix while not affecting the prefix or suffix. In the case of rewriting, the model also preserved the meaning of the original text. 

The Hugging Face repo was used to access pre-trained GPT-2 models along with an existing dataset. An additional dataset was created to train and evaluate the model fine tuned for rewriting. 

The metrics for accuracy was a combination of human evaluation (for determing whether a successful insert/rewrite was performed) and numerical analysis (counting how many words were generated compared to the original text). The evaluation set consisted of 20 data points.

The insertion model was able to achieve an accuracy of 75% compared to 50% accuracy of the baseline model while the rewriting model achieved an accuracy of 55% compared to 25% for the baseline. 

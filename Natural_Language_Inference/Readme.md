Natural Language Inference

Task is to infer from two sentence pair that they represent entailment, contradiction or neutral (Not correlated). Task becomes bit challenging for model because it is required to identify most important words in both sentences and their relation.  Here I have solved this task by two approaches and compared results.
1. Logistic regression classiﬁer using TF-IDF features
2. Deep model using Bidirection LSTM with sentence fusion

I have tested both approaches on one of the standered data set Stanford Natural Language Inference (SNLI). I am getting test accuracy of 64.37% and 82.17% for LR model and Deep model respectively.
Results clearly states that Deep approach outperform LR based approach with great margin that is due to ability of LSTM model to capture sequential details.
More details regarding models, reasoning and results can be found in report.
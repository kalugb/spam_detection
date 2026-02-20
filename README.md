A simple TFIDF + NN SMS classification using PyTorch

Dataset info:
957 rows, 122 Spam (Class 1), 835 Non-Spam (Class 0)

Dataset obtained from Kaggle, dataset total rows aren't much so the test results might not reflect actual model performance in real-world detection, but just a simple project that can be expanded later
Can use HuggingFace DistilBERT for this classification as well but that might be overkill for very small dataset like this

Test results:

Total test rows: 125 (very small)
Non-Spam (Class 0): 49, Spam (Class 1): 76

Info (Obtained from python terminal using classification_report in sklearn):

Class 0 (Non-Spam)    
Precision: 0.87, Recall: 0.98, F1: 0.92

Class 1 (Spam)
Precision: 0.99, Recall: 0.91, F1: 0.95

Accuracy: 0.94

Confusion Matrix (Obtained from python terminal using confusion_matrix in sklearn):
[[48  1]
 [7  69]]
 TP: 48, FP: 1, FN: 7, TN: 69 

Feel free to try using other dataset to test the model as well, make sure the test dataset must be in this format: (Message_body, Label)

Anyway thanks for reading and try out this simple project

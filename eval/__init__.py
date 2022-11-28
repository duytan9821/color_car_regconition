import torch
from sklearn.metrics import classification_report
from utils import read_json
from configs import *

def eval(model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        device: str='cpu'):   
    """
        > Evaluate the model on the given dataset and return the predicted labels and true labels
        
        :param model: the model to be evaluated
        :type model: torch.nn.Module
        :param dataloader: The DataLoader object that contains the data to be evaluated
        :type dataloader: torch.utils.data.DataLoader
        :param device: The device to run the model on, defaults to cpu
        :type device: str (optional)
    """
    # Put model in eval mode
    model.eval() 
    # Turn on inference context manager
    all_predicts = []
    all_labels = []
    # Loop through DataLoader batches
    for _ , batch in enumerate(dataloader):
        # Send data to target device
        batch_images, batch_labels = batch[0].to(device), batch[1].to(device)

        # 1. Forward pass
        test_pred_logits = model(batch_images)

        # Calculate and accumulate accuracy
        test_pred = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
        all_predicts.extend(test_pred.cpu().numpy().tolist())
        all_labels.extend(batch_labels.cpu().numpy().tolist())
    all_labels_tags = [IDX2TAG[str(idx)] for idx in  all_labels]
    all_predicts_tags = [IDX2TAG[str(idx)] for idx in  all_predicts]
    print(classification_report(y_true=all_labels_tags, y_pred=all_predicts_tags))
    return all_predicts, all_labels

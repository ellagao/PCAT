import torch
import math
from config import Config

def mapping(labels):
    for idx in range(labels.size()[0]):
        labels[idx] = Config.label_map[labels[idx]]
    return labels

def inv_mapping(pred):
    for idx in range(pred.size()[0]):
        pred[idx] = Config.label_inv_map[pred[idx]]
    return pred

def test(dataloader, model1, model2=None, model3=None, label_shuf_flag=False, device = Config.device):
    '''
    model1, model2, model3 = whole_model, None, None
    model1, model2, model3 = server, client, None
    model1, model2, model3 = top, server, bottom
    It doesn't contain the case: model2 == None and model3 != None.
    '''
    assert not(model2 == None and model3 != None), "We use model2 first instead of model3!"

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if Config.criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
        
    model1.eval()
    if model2 != None: model2.eval()
    if model3 != None: model3.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if model2 == None and model3 == None:
                pred = model1(images)
            elif model2 != None and model3 == None:
                pred = model1(model2(images))
            elif model2 != None and model3 != None:
                pred = model1(model2(model3(images)))
            test_loss += criterion(pred, labels).item()
            
            if label_shuf_flag == True:
                pred = inv_mapping(pred.argmax(1))
            else:
                pred = pred.argmax(1)

            correct += (pred == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return 100.0 * correct, test_loss

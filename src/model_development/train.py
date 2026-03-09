import torch
import pandas as pd
from src.model_development import architectures as model_module
from sklearn.metrics import roc_auc_score


def evaluate_loss(model, data_loader, loss_criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    labels_list = []
    predicts_list = []
    with torch.no_grad():
        for data in data_loader:
            labels = data['labels'].long()
            outputs = model(data['features'])
            loss = loss_criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1
            labels_list.extend(labels.cpu().numpy())
            predicts_list.extend(outputs[:, 1].cpu().numpy())

    auc = roc_auc_score(labels_list, predicts_list)
    model.train()
    return total_loss / max(num_batches, 1), auc


def get_early_stopping_name(**kwargs):
    if kwargs.get('early_stopping', None):
        early_params = {} if isinstance(kwargs['early_stopping'], bool) else kwargs['early_stopping']
        metric = early_params.get('metric', 'loss')
        patience = early_params.get('patience', None)
        min_epochs = early_params.get('min_epochs', 1)
        return f"-earlystop-{metric}-patience_{patience}-minEpochs_{min_epochs}"
    return "-no_earlystop"


def train(train_loader, val_loader, in_channels, amount_classes, model_params, indent=0):
    model_name = model_params['name']
    model_class = getattr(model_module, model_name)
    
    model_params['in_channels'] = in_channels
    model_params['num_classes'] = amount_classes
    model = model_class(**model_params)
    model.to(model_params['model_device'])
    
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get('learning_rate', 0.0001))
    loss_tracker = []
    model.train()
    
    early_stopping = model_params.get('early_stopping', None)
    if early_stopping:
        early_params = {} if isinstance(model_params['early_stopping'], bool) else model_params['early_stopping']
        patience = early_params.get('patience', None)
        metric = early_params.get('metric', 'loss')
        min_epochs = early_params.get('min_epochs', 1)
        prev_loss, prev_auc = evaluate_loss(model, val_loader, loss_criterion)
        loss_tracker.append({'split': 'val', 'epoch': 0, 'loss': prev_loss, 'auc': prev_auc})
        counter, prev_stat = 0, None
        best_metric_value, best_epoch = prev_loss if metric == 'loss' else prev_auc, 1
    else: 
        best_epoch = None

    for epoch in range(model_params['epochs']): 
        epoch_loss = 0.0
        num_batches = 0
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            labels = data['labels'].long()
            outputs = model(data['features'])

            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        train_loss = epoch_loss / max(num_batches, 1)
        loss_tracker.append({'split': 'train', 'epoch': epoch+1, 'loss': train_loss})
    
        if early_stopping:
            val_loss, val_auc = evaluate_loss(model, val_loader, loss_criterion)
            loss_tracker.append({'split': 'val', 'epoch': epoch+1, 'loss': val_loss, 'auc': val_auc})
            
            if epoch+1 >= min_epochs:
                if patience:
                    if patience == counter + 1 and ((metric == 'loss' and val_loss >= best_metric_value) or 
                                                    (metric == 'auc' and val_auc <= best_metric_value)):
                        break
                    
                    elif metric == 'loss' and val_loss < best_metric_value:
                        best_metric_value = val_loss
                        best_epoch = epoch+1
                        counter = 0
                    elif metric == 'auc' and val_auc > best_metric_value:
                        best_metric_value = val_auc
                        best_epoch = epoch+1
                        counter = 0
                    else:
                        counter += 1

                elif (metric == 'loss' and val_loss > prev_stat[0]) or (metric == 'auc' and val_auc < prev_stat[1]):
                    best_epoch = prev_stat[2]
                    break
            prev_stat = (val_loss, val_auc, epoch+1)
    return model, pd.DataFrame(loss_tracker), best_epoch

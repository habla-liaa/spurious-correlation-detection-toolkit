import numpy as np
import torch


def test(model, test_loader):
    model.eval()
    test_labels, test_predict, ids, groups = [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            test_labels.append(data['labels'].cpu().numpy())

            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(model(data['features'])).cpu().detach().numpy()
            test_predict.append(probabilities)

            ids.append(data['ids'])
            groups.append(data['groups'])

    return [np.concatenate(test_labels), np.concatenate(test_predict), np.concatenate(groups), np.concatenate(ids)]

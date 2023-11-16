import torch 
import numpy

def classification_report(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = (y_true == y_pred).mean()

    num_classes = len(set(y_true))
    confusion_matrix = numpy.zeros(num_classes, num_classes)

    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1
    
    precision = numpy.zeros(num_classes)
    recall = numpy.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    for i in range(num_classes):
        true_positive = confusion_matrix[i, i]
        false_positive = confusion_matrix[:, i].sum() - true_positive
        false_negative = confusion_matrix[i, :].sum() - true_positive

        if true_positive + false_positive != 0:
            precision[i] = true_positive / (true_positive + false_positive)
        else:
            precision[i] = 0

        if true_positive + false_negative != 0:
            recall[i] = true_positive / (true_positive + false_negative)

        if precision[i] + recall[i] != 0:
            f1[i] = 2*(precision[i]*recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0

    print("Accuracy: {:.4f}".format(accuracy))
    print("\nClass-wise Metrics:")
    print("{:<15} {:<15} {:<15} {:<15}".format("Class", "Precision", "Recall", "F1 Score"))

    for i in range(num_classes):
        class_name = f"Class {i}"
        print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format(class_name, precision[i], recall[i], f1[i]))

def train(model, criterion, optimizer, dataloader, device):
    epoch_loss = 0
    predicted_labels = []
    true_labels = []

    for idx, (labels, inputs) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels.float())
        epoch_loss += loss
        print(f'itteration {idx}, {loss}') ## remove
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).sum(axis=1).cpu().numpy()

        predicted_labels.extend(outputs)
        true_labels.extend(labels.squeeze().cpu().numpy())

        loss.backward()
        optimizer.step()

    return true_labels, predicted_labels
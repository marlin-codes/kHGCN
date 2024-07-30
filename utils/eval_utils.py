from sklearn.metrics import average_precision_score, accuracy_score, f1_score


def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


def curv_acc(output, label, curvature):
    prediction = output.max(1)[1].type_as(label)
    # prediction=label
    if prediction.is_cuda:
        prediction = prediction.cpu()
        label = label.cpu()

    n = prediction.shape[0]
    neg_acc = 0
    zero_acc = 0
    pos_acc = 0

    # Loop and increment counters
    for i in range(n):
        if label[i] == prediction[i]:
            try:
                if -2 <= curvature[i] <= -0.01:
                    neg_acc += 1
                elif -0.01 < curvature[i] <= 0.1:
                    zero_acc += 1
                elif 0.1 < curvature[i] <= 2:
                    pos_acc += 1
            except:
                pass

    # Calculate percentages
    neg_acc_perc = neg_acc / n
    zero_acc_perc = zero_acc / n
    pos_acc_perc = pos_acc / n

    return neg_acc_perc, zero_acc_perc, pos_acc_perc

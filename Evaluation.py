#confusion matrix, accuracy , precision, recall, F1 score

def confusion_matrix (label, predicted_value):
    """ takes label, predicted_value as vectors
        returns confusion matrix , tp, fp, tn, fn """



def accuracy (label, predicted_value):
    """takes label , predicted_value as vectors
        return accuracy """

    _, tp, fp, tn, fn = confusion_matrix(label, predicted_value)
    Accuracy = (tp+tn) / (tp+fp+tn+fn)
    return Accuracy


def precision (label, predicted_value):
    """takes label , predicted_value as vectors
        return precision """

    _, tp, fp, tn, fn = confusion_matrix(label, predicted_value)
    Precision = tp / tp + fp
    return Precision


def recall (label, predicted_value):
    """takes label , predicted_value as vectors
        return recall """

    _, tp, fp, tn, fn = confusion_matrix(label, predicted_value)
    Recall = tp / (tp + fn)
    return Recall


def F1_score(label, predicted_value):
    """takes label , predicted_value as vectors
        return F1_score """

    Precision = precision(label, predicted_value)
    Recall = recall(label, predicted_value)
    F1 = 2 * (Precision * Recall)/(Precision + Recall)
    return F1
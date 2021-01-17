#confusion matrix, accuracy , precision, recall, F1 score

def confusion_matrix (label, predicted_value, num_class):
    """ takes label, predicted_value as vectors
        returns confusion matrix , tp, fp, tn, fn """
    classes = list(range(num_class))


    #initialize key,values  2d array
    for i in classes:
        classes[i] = [0] * num_class

    for i in range(len(label)):
        classes[label[i]][predicted_value[i]] += 1

    return classes

    


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


def print_2dlist(arr):

    idx = 0
    l = list(range(len(arr)))

    print(str('    ') + str(l))
    for i in arr:
        print(str(idx) +'-->' + str(i))
        idx +=1


if __name__ == '__main__':

    label =           [0,1,2,3,4,5,1]
    predicted_value = [0,1,2,4,5,3,1]

    out = confusion_matrix(label, predicted_value, 6)
    print_2dlist(out)

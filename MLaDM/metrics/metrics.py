
import numpy as np


def unique_labels(*list_of_labels):
    
    list_of_labels = [np.unique(labels[np.isfinite(labels)].ravel())
                      for labels in list_of_labels]
    list_of_labels = np.concatenate(list_of_labels)
    return np.unique(list_of_labels)


def confusion_matrix(y_true, y_pred, labels=None):
   
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    n_labels = labels.size

    CM = np.empty((n_labels, n_labels), dtype=np.long)
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            CM[i, j] = np.sum(
                np.logical_and(y_true == label_i, y_pred == label_j))

    return CM


def roc_curve(y, probas_):
    
    y = y.ravel()
    probas_ = probas_.ravel()
    thresholds = np.sort(np.unique(probas_))[::-1]
    n_thresholds = thresholds.size

    tpr = np.empty(n_thresholds) # True positive rate
    fpr = np.empty(n_thresholds) # False positive rate
    n_pos = float(np.sum(y == 1)) # nb of true positive
    n_neg = float(np.sum(y == 0)) # nb of true negative

    for i, t in enumerate(thresholds):
        tpr[i] = np.sum(y[probas_ >= t] == 1) / n_pos
        fpr[i] = np.sum(y[probas_ >= t] == 0) / n_neg

    return fpr, tpr, thresholds


def auc(x, y):
   
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    h = np.diff(x)
    area = np.sum(h * (y[1:] + y[:-1])) / 2.0
    return area


def precision_score(y_true, y_pred, pos_label=1):
   
    p, _, _, s = precision_recall_fscore_support(y_true, y_pred)
    if p.shape[0] == 2:
        return p[pos_label]
    else:
        return np.average(p, weights=s)


def recall_score(y_true, y_pred, pos_label=1):
  
    _, r, _, s = precision_recall_fscore_support(y_true, y_pred)
    if r.shape[0] == 2:
        return r[pos_label]
    else:
        return np.average(r, weights=s)


def fbeta_score(y_true, y_pred, beta, pos_label=1):
    
    _, _, f, s = precision_recall_fscore_support(y_true, y_pred, beta=beta)
    if f.shape[0] == 2:
        return f[pos_label]
    else:
        return np.average(f, weights=s)


def f1_score(y_true, y_pred, pos_label=1):
   
    return fbeta_score(y_true, y_pred, 1, pos_label=pos_label)


def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None):
   
    assert(beta > 0)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    n_labels = labels.size
    true_pos = np.zeros(n_labels, dtype=np.double)
    false_pos = np.zeros(n_labels, dtype=np.double)
    false_neg = np.zeros(n_labels, dtype=np.double)
    support = np.zeros(n_labels, dtype=np.long)

    for i, label_i in enumerate(labels):
        true_pos[i] = np.sum(y_pred[y_true == label_i] == label_i)
        false_pos[i] = np.sum(y_pred[y_true != label_i] == label_i)
        false_neg[i] = np.sum(y_pred[y_true == label_i] != label_i)
        support[i] = np.sum(y_true == label_i)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    precision[(true_pos + false_pos) == 0.0] = 0.0
    recall[(true_pos + false_neg) == 0.0] = 0.0

    beta2 = beta ** 2
    fscore = (1 + beta2) * (precision * recall) / (
        beta2 * precision + recall)

    fscore[(precision + recall) == 0.0] = 0.0

    return precision, recall, fscore, support


def classification_report(y_true, y_pred, labels=None, class_names=None):
   
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    last_line_heading = 'avg / total'

    if class_names is None:
        width = len(last_line_heading)
        class_names = ['%d' % l for l in labels]
    else:
        width = max(len(cn) for cn in class_names)
        width = max(width, len(last_line_heading))


    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels)
    for i, label in enumerate(labels):
        values = [class_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["%0.2f" % float(v)]
        values += ["%d" % int(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["%0.2f" % float(v)]
    values += ['%d' % np.sum(s)]
    report += fmt % tuple(values)
    return report


def precision_recall_curve(y_true, probas_pred):
   
    y_true = y_true.ravel()
    labels = np.unique(y_true)
    if np.all(labels == np.array([-1, 1])):
        # convert {-1, 1} to boolean {0, 1} repr
        y_true[y_true == -1] = 0
        labels = np.array([0, 1])
    if not np.all(labels == np.array([0, 1])):
        raise ValueError("y_true contains non binary labels: %r" % labels)

    probas_pred = probas_pred.ravel()
    thresholds = np.sort(np.unique(probas_pred))
    n_thresholds = thresholds.size + 1
    precision = np.empty(n_thresholds)
    recall = np.empty(n_thresholds)
    for i, t in enumerate(thresholds):
        y_pred = np.ones(len(y_true))
        y_pred[probas_pred < t] = 0
        p, r, _, _ = precision_recall_fscore_support(y_true, y_pred)
        precision[i] = p[1]
        recall[i] = r[1]
    precision[-1] = 1.0
    recall[-1] = 0.0
    return precision, recall, thresholds


def explained_variance_score(y_true, y_pred):
    
    return 1 - np.var(y_true - y_pred) / np.var(y_true)


def r2_score(y_true, y_pred):
   
    return 1 - (((y_true - y_pred)**2).sum() /
                ((y_true - y_true.mean())**2).sum())


def zero_one_score(y_true, y_pred):
   
    return np.mean(y_pred == y_true)


###############################################################################
# Loss functions

def zero_one(y_true, y_pred):
  
    return np.sum(y_pred != y_true)


def mean_square_error(y_true, y_pred):
   
    return np.linalg.norm(y_pred - y_true) ** 2

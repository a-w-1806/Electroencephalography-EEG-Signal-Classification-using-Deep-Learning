from sklearn.metrics import confusion_matrix
import numpy as np

def print_all_stats(y_true, y_pred):
    cmx = confusion_matrix(y_true, y_pred)
    tn = cmx[0][0]
    tp = cmx[1][1]
    fp = cmx[0][1]
    fn = cmx[1][0]

    precision = tp / (tp + fp)
    recall = tp / (tp + tn)
    silence = 1 - recall
    noise = 1 - precision
    error = (fp + fn) / (tp + fn)
    f1 = 2 * (recall * precision) / (precision + recall)
    reco = (tp + tn) / (tp + tn + fp + fn)

    print("TP:{}\tTN:{}\tFP:{}\tFN:{}\n".format(tp, tn, fp, fn))
    print("Reco:{}\tRecall:{}\tPrecision:{}\tSilence:{}\n".format(reco, recall, precision, silence))
    print("Noise:{}\tError:{}\tF-measure:{}".format(noise, error, f1))

    return {"tn":tn,"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,
            "silence":silence,"noise":noise,"error":error,"f1":f1,"reco":reco}

def Information_Transfer_Rate(P, num_aggregate, num_classes = 36):
    """
    ITR
    :param P: the probability to recognize a character
    :param num_aggregate: 1-15
    :param num_classes:
    :return: float
    """
    return 60*(P * np.log2(P) + (1 - P)*np.log2((1-P)/(num_classes-1))+np.log2(num_classes))/(2.5+2.1*num_aggregate)



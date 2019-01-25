from src.metrics import PAMI
import numpy as np

def test_print_all_stats():
    y_true = np.array([1,0,0,1,0,0,1,1,1,0])
    y_pred = np.array([1,0,0,1,1,0,1,0,1,1])
    stats = PAMI.print_all_stats(y_true,y_pred)
    assert stats['tp'] == 4 and stats['tn'] == 3 and stats['fp'] == 2 and stats['fn'] == 1
    assert stats['recall'] == 4/(4+3) and stats['precision'] == 4/(4+2)
    assert stats['silence'] == 1-stats['recall'] and stats['noise'] == 1-stats['precision']
    assert stats['error'] == (stats['fp']+stats['fn'])/(stats['tp']+stats['fn'])
    assert stats['f1'] == 2*(stats['recall']*stats['precision'])/(stats['precision']+stats['recall'])
    assert stats['reco'] == (stats['tp']+stats['tn'])/(stats['tp']+stats['tn']+stats['fp']+stats['fn'])
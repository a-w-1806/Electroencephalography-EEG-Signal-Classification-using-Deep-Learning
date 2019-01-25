import src.data.preprocessing
from src import pipelines
import numpy as np
import pytest

def test_code_reorder():
    probs = np.random.random([25,15,12])
    code = np.zeros([25,15,12])                       # code 1-12
    for i in range(25):
        for j in range(15):
            code[i,j] = np.random.permutation(12)+1

    reorderd = pipelines.code_reorder(probs,code)

    for i in range(25):
        for j in range(15):
            for k in range(12):
                assert reorderd[i,j,k] == probs[i,j,np.argwhere(code[i,j] == k+1).squeeze()]


@pytest.mark.parametrize("num_aggregate",[1,5,12])
def test_aggregate(num_aggregate):
    sort = np.random.random([25,15,12])
    aggregated = pipelines.aggregate_prob_across_trials(sort,num_aggregate)  # [25,12]

    for i in range(25):
        for j in range(12):
            prob_accu = 0
            for k in range(num_aggregate):
                prob_accu += sort[i,k,j]
            assert prob_accu == aggregated[i,j]


@pytest.mark.parametrize("prob,rowcols",[
    (np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95],[0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.2,0.1]])
,np.array([[5,5],[0,0]])),
    (np.array([[0.3,0.2,0.1,0.6,0.2,0.1,0.1,0.97,0.8,0.75,0.2,0.94],[0.95,0.99,0.23,0.47,0.56,0.89,0.27,0.59,0.29,0.50,0.45,0.39]])
,np.array([[3,1],[1,1]]))])
def test_prob_to_rowcols(prob,rowcols):
    assert (rowcols == pipelines.prob_to_rowcols(prob)).all()  # (25,2)


@pytest.mark.parametrize("paradigm",[np.array([['A','B','C','D','E','F'],['G','H','I','J','K','L'],['M','N','O','P','Q','R'],
                     ['S','T','U','V','W','X'],['Y','Z','1','2','3','4'],['5','6','7','8','9','_']])])
@pytest.mark.parametrize("data,letters",[(np.array([[2,4],[3,1]]),np.array(['1','J'])),
                                         (np.array([[0,0],[5,5]]),np.array(['A','_'])),
                                         (np.array([[5,0],[0,5]]),np.array(['F','5']))])
def test_letter_lookup(data,paradigm,letters):
    assert (pipelines.letter_lookup(data,paradigm) == letters).all()


@pytest.mark.parametrize("letters,target_string,acc",[(np.array(['a','b','c']),"abc",1),
                                                      (np.array(['a','c','e']),"cae",1/3),
                                                      (np.array(['w','e','r']),"abc",0),
                                                      (np.array(['f','g','h']),"fgq",2/3)])
def test_accuracy(letters,target_string,acc):
    assert pipelines.accuracy(letters,target_string) == acc
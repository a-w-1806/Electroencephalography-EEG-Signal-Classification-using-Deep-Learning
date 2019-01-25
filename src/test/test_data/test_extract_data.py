from src.data import extract_data
import numpy as np
import pytest

# @pytest.mark.parametrize("path",["Subject_A_Train","Subject_B_Train"])
# @pytest.mark.parametrize("seconds_to_slice",[1.0])
# def test_extract_data_test(path,seconds_to_slice):
#     data_dir = "./data/raw/BCI_Comp_III_Wads_2004/"
#     raw = extract_data.read_BCIIII_p300_mat(data_dir+path)
#     extracted = extract_data.extract_data(data_dir+path,seconds_to_slice)
#     # Let's go through every corner
#     for letter in range(85):
#         timepoints = extract_data.find_timepoints_1D(raw['StimulusCode'][letter,:])
#         assert len(timepoints) == 12 * 15
#         for time in range(15):
#             assert extracted['code'][letter,time].sum() == 1+2+3+4+5+6+7+8+9+10+11+12
#             assert extracted['label'][letter,time].sum() == 1+1
#             for inten in range(12):
#                 startpoint = timepoints[time*12+inten]
#                 for channel in range(64):
#                     assert np.all(extracted['signal'][letter,time,inten,channel,:] == raw['Signal'][letter,startpoint:(startpoint+int(seconds_to_slice  *240)),channel])
#                     assert extracted['code'][letter,time,inten] == raw['StimulusCode'][letter,startpoint]
#                     assert extracted['label'][letter,time,inten] == raw['StimulusType'][letter,startpoint]
#                     assert extracted['label'].sum() == 30*85



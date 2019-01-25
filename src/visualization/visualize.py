import numpy as np
import matplotlib.pyplot as plt

def electrodes_selected_visualize(signal,elec_to_vis,n_rows,x_vline = None):
    """
    Visualize some selected electrodes

    :param signal: (electrodes,time)
    :param elec_to_vis: [num_1,num_2,...]
    """
    plt.figure()
    n_cols = np.ceil(len(elec_to_vis)/n_rows)
    for num,electrode in enumerate(elec_to_vis):
        plt.subplot(n_rows,n_cols,num+1)
        plt.plot(signal[electrode])
        if x_vline:
            plt.axvline(x_vline, linewidth=1, linestyle=':', color='coral')

    plt.show()
    return
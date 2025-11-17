import numpy as np
import scipy.special as sp
import math


def significance(B):
    """
    Welbanks+, 2021
    :param B: Bayes factor Z1/Z2
    :return DS: Detection significance
    """
    DS = np.sqrt(2) * 1 / (sp.erfc(np.exp(sp.lambertw((-1 / (B * math.e)), k=-1))))

    return DS


evidence1 = -5.043e+07   # model 1 -> A0, Abeam+refl, Aellip, T0
evidence2 = -50842137.910   # model 2 -> A0, Abeam+refl, T0



# Bayes factor computation
bayes_factor = evidence1 / evidence2

if bayes_factor >= 1:
    print('Bayes factor is: ', bayes_factor)
    print("Model 1 is better than model 2.")
    # Sigma computation
    sigma = significance(bayes_factor)
    print(f"Model 1 is favoured over model 2 by {np.round(sigma.real, 1)} sigma.")

else:
    print('Bayes factor is: ', bayes_factor)
    print("Model 2 is better than model 1.")
    sigma = significance(1 / bayes_factor)
    print(f"Model 2 is favoured over model 1 by {np.round(sigma.real, 1)} sigma.")

import scipy.stats as ss
import pandas as pd
import numpy as np

def cramers_corrected_stat(x, y):
    """Calculate Cramers V statistic for categorical-categorical association.
        Uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328

    Args:
        x (Series): First categorical variable.
        y (Series): Second categorical variable.
    
    Returns:
        float: Cramers V statistic.
    """
    result = -1
    if len(x.value_counts()) == 1:
        print("First variable is constant")
    elif len(y.value_counts()) == 1:
        print("Second variable is constant")
    else:
        conf_matrix = pd.crosstab(x, y)
        
        if conf_matrix.shape[0] == 2:
            correct = False
        else:
            correct = True
        
        chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]
        
        n = sum(conf_matrix.sum())
        phi2 = chi2 / n
        r, k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        result = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return round(result, 6)

from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
15552000
[7.26785044e-01 2.70025592e-01 2.97839506e-03 2.10776749e-04
 1.92901235e-07]
15552000
[9.81738233e-01 1.78772505e-02 3.81687243e-04 2.82921811e-06]
15552000
[9.71521155e-01 2.77515432e-02 6.44483025e-04 8.15329218e-05
 1.28600823e-06]
'''
# UM
a = [11302961, 4199438, 46320, 3278, 3]
print(np.sum(a))
print(a / np.sum(a))
a = [15267993, 278027, 5936, 44]
print(np.sum(a))
print(a / np.sum(a))

a = [15109097, 431592, 10023, 1268, 20]
print(np.sum(a))
print(a / np.sum(a))

data = pd.DataFrame(
    {
    'Solver': ['UM', 'UM', 'UM', 'UM', 'UM', 'C2I', 'C2I', 'C2I', 'C2I', 'C2G', 'C2G', 'C2G', 'C2G', 'C2G'],
    'Times Covered': [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3, 4],
    'Counts': [11302961, 4199438, 46320, 3278, 3, 15267993, 278027, 5936, 44, 15109097, 431592, 10023, 1268, 20]
    }
)

sns.barplot(data=data, x='Times Covered', y='Counts', hue='Solver')
plt.savefig('distri.eps', bbox_inches = 'tight')
plt.show()
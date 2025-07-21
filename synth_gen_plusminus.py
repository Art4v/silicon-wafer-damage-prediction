# import libraries
import pandas as pd
import numpy as np

# set random seed for reporducibility
np.random.seed(0)

# define criteria
criteria = {
    "+Criteria 1": (1.0, 1.0),
    "+Criteria 2": (2.0, 1.0),
    "+Criteria 3": (3.0, 1.0),
    "+Criteria 4": (4.0, 1.0),
    "-Criteria 5": (5.0, 1.0),
    "-Criteria 6": (6.0, 1.0),
    "-Criteria 7": (7.0, 1.0),
    "-Criteria 8": (8.0, 1.0),
}

# number of samples
n = 100

# prepare data storage
data = []
criteria_list = list(criteria)

#generate synthetic data
for i in range (n):
    sample_no = i + 1
    params = criteria_list
    row = {
        'Sample No.': sample_no,
        '+Criteria 1': np.random.normal(*criteria_list['Criteria 1'])
    }
    data.append(row)

print(pd.DataFrame(data))


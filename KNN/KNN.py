import numpy as np
import pandas as pd
import myUtil

# preprocess
X,Y = myUtil.preprocess1(*myUtil.read_data("train.csv"))

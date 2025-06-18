import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import importlib
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, f1_score,
    accuracy_score, confusion_matrix, hamming_loss, zero_one_loss
)
from sklearn.model_selection import (
    train_test_split, cross_val_predict, StratifiedKFold,
    cross_val_score, KFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer,MinMaxScaler

from sklearn.multioutput import MultiOutputClassifier

from tslearn.clustering import TimeSeriesKMeans

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Reshape

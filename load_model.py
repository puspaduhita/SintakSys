!pip install docx2txt
!pip install tensorflow==2.5.0

import os
import re
import numpy as np
import docx2txt

from keras.models import Model, load_model
from keras.layers import Input

np.random.seed(1234)

SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.
CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
REMOVE_CHARS = '[#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'


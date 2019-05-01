'''
select patient

configure data handler

load model

for each slide in scan
    
    generate patches

    feed patches into model

    create binary slide


'''

import keras
import model_lib as ml
import data_handler as dh
import logger_lib as ll
import numpy as np
import os

'''
Authors: Jeff Adrion, Andrew Kern, Jared Galloway
'''

from ReLERNN.imports import *

def GRU_TUNED84(x,y):
    '''
    Same as GRU_VANILLA but with dropout AFTER each dense layer.
    '''
    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = Input(shape=(numSNPs,numSamps))
    model = layers.Bidirectional(layers.CuDNNGRU(84,return_sequences=False))(genotype_inputs)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    #----------------------------------------------------

    position_inputs = Input(shape=(numPos,))
    m2 = Dense(256)(position_inputs)

    #----------------------------------------------------


    model =  layers.concatenate([model,m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    #----------------------------------------------------

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def GRU_TUNED84_NORM(x,y):
    '''
    Same as GRU_VANILLA but with dropout AFTER each dense layer.
    '''
    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = Input(shape=(numSNPs,numSamps))
    model = layers.Bidirectional(layers.CuDNNGRU(84,return_sequences=False))(genotype_inputs)
    model = layers.BatchNormalization()
    model = layers.Dense(256)(model)
    model = layers.BatchNormalization()
    model = layers.Dropout(0.35)(model)

    #----------------------------------------------------

    position_inputs = Input(shape=(numPos,))
    m2 = Dense(256)(position_inputs)

    #----------------------------------------------------


    model =  layers.concatenate([model,m2])
    model = layers.Dense(64)(model)
    model = layers.BatchNormalization()
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    #----------------------------------------------------

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def GRU_TUNED84_S2S(x,y):
    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = Input(shape=(numSNPs,numSamps))
    position_inputs = Input(shape=(numPos,))
    lstm1, state_h, state_c = LSTM(numSNPs, return_state=True)(genotype_inputs) # state_h is the sequence corresponding to each input timestep

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[state_h])
    model.compile(optimizer='Adam', loss='mae', metrics=["accuracy"])
    model.summary()

    return model

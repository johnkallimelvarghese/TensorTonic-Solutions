import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x=np.array(x,dtype=np.float64)
    W1=np.array(W1,dtype=np.float64)
    W2=np.array(W2,dtype=np.float64)
    Ws=np.array(Ws,dtype=np.float64)

    main=x @ W1
    main=np.maximum(0,main)

    main= main @ W2

    shortcut= x @ Ws

    y=np.maximum(0,main+shortcut)

    return np.round(y,4).tolist()
    
import numpy as np

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    """
    Returns updated classifier parameters and the pre-update loss.
    """
    #cls state extraction
    X=hidden_states[:,0,:]

    #leniar classification
    Z=X @ classifier_W + classifier_b

    #softmax/ classifications into probabilities
    z_stable=Z-np.max(Z,axis=1,keepdims=True)
    exp_z=np.exp(z_stable)
    P=exp_z/np.sum(exp_z,axis=1,keepdims=True)

    #loss calculation
    B=X.shape[0]
    correct_prob=P[np.arange(B),labels]
    loss=-np.mean(np.log(correct_prob))

    #gradiant/ backpropagation dZ=P-Y/B
    Y=np.zeros_like(P)
    Y[np.arange(B),labels]=1
    dz=(P-Y)/B
    dw=X.T @ dz
    db=np.sum(dz,axis=0,)

    #gradient descent
    new_classifier_W=classifier_W-learning_rate*dw
    new_classifier_b=classifier_b-learning_rate*db

    return{
        "new_classifier_W":np.asarray(new_classifier_W,dtype=np.float64),
        "new_classifier_b":np.asarray(new_classifier_b,dtype=np.float64),
        "loss":loss
    }
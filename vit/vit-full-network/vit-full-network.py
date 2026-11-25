import numpy as np

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    B,H,W,C_in=image.shape
    P=patch_size

    H_c=(H//P)*P
    W_c=(W//P)*P
    image = image[:, :H_c, :W_c, :]

    num_h=H_c//P
    num_w=W_c//P
    N=num_h*num_w
    patches=image.reshape(
        B,
        num_h,
        P,
        num_w,
        P,
        C_in
    )
    patches=patches.transpose(0,1,3,2,4,5)
    patches=patches.reshape(B,N,P*P*C_in)

    embeddings= patches @ W_patch + patch_bias

    cls_token=np.repeat(cls_token,B,axis=0)
    embeddings=np.concatenate([cls_token,embeddings],axis=1)
    embeddings=embeddings+pos_embed
    for weights in encoder_weights:
        #1st layer normalization
        mean=np.mean(embeddings,axis=-1,keepdims=True)
        variance=np.var(embeddings,axis=-1,keepdims=True)
        x_dim=(embeddings-mean)/np.sqrt(variance+1e-6)

        Q=x_dim @ weights["Wq"]
        K=x_dim @ weights["Wk"]
        V=x_dim @ weights["Wv"]

        D=Q.shape[-1]
        head_dim=D//num_heads

        Q=Q.reshape(B,N+1,num_heads,head_dim)
        K=K.reshape(B,N+1,num_heads,head_dim)
        V=V.reshape(B,N+1,num_heads,head_dim)

        Q=Q.transpose(0,2,1,3)
        V=V.transpose(0,2,1,3)

        scores=Q @ K.transpose(0,2,3,1)
        scores=scores/np.sqrt(head_dim)

        #softmax
        scores=scores-np.max(scores, axis=-1, keepdims=True)
        attention=np.exp(scores)
        attention=attention/np.sum(attention, axis=-1, keepdims=True)

        #weighted sum
        context=attention @ V

        #combine attention heads
        context=context.transpose(0,2,1,3)
        context=context.reshape(B,N+1,D)

        attention_output= context @ weights["Wo"]

        #1st residual connection
        embeddings=embeddings + attention_output

        #2nd Layer normalization
        mean=np.mean(embeddings,axis=-1,keepdims=True)
        variance=np.var(embeddings, axis=-1,keepdims=True)
        x_norm = (embeddings - mean) / np.sqrt(variance + 1e-6)

        #mlp 1st leniar layer
        mlp_hidden= x_norm @ weights["W1"]

        #GELU
        mlp_hidden = 0.5 * mlp_hidden * (
            1 + np.tanh(
                np.sqrt(2 / np.pi) *
                (mlp_hidden + 0.044715 * mlp_hidden**3)
            )
        )

        #MLP 2nd leniar layer
        mlp_output=mlp_hidden @ weights["W2"]

        #2nd residual connection
        embeddings=embeddings+mlp_output

    #final normalization before classification
    mean=np.mean(embeddings,axis=-1,keepdims=True)
    variance=np.var(embeddings,axis=-1,keepdims=True)
    embeddings=(embeddings-mean)/np.sqrt(variance+1e-6)

    #CLS token extraction
    cls_token=embeddings[:,0,:]

    logits= cls_token @ W_head

    return logits.astype(np.float64)

        

        
    
"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x, dtype=float)
    for i in np.ndindex(x.shape):
        orig = x[i]
        x[i] = orig + eps
        f_plus = f(x)
        x[i] = orig-eps
        f_minus = f(x)
        grad[i] = (f_plus-f_minus)/(2.0*eps)
        x[i] = orig
        
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    analytic_grad_array = np.asarray(analytic_grad)
    numeric_grad_array = np.asarray(numeric_grad, dtype=float)
    diff = np.abs(analytic_grad_array - numeric_grad_array) # error difference, elemntwise
    max_analytic_error = np.max(analytic_grad_array)
    max_numeric_error = np.max(numeric_grad)

    max_error = np.maximum(max_analytic_error, max_numeric_error)
    max_error = np.maximum(max_error, tol)
    relative_error = diff/ max_error
    
    return np.max(relative_error)

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    
    W,b = weight_init_fn(in_dim, out_dim) 
    params = {'W': W, 'b': b}
    
    def forward(x): # x -> (batch, in_dim)
      # comptues the forward pass
      y = x @ params['W'] + params['b'] # y -> (batch, out_dim)
      cache = x
      return y, cache

    def backward(dout, cache): # dout: (batch, out_dim), cache <- x of the current layer 
      # computes backprop
      '''
        analytic gradients are:
        dout = dL/dy
        dx = dL/dx = dL/dy * dy/dx
        dW = dL/dW = dL/dy * dy/dW 
        db = dL/db= dL/dy * dy/db = 
      '''
      x = cache
      # dx: (batch, in_dim)
      # dW: (in_dim, out_dim)
      # db: (out_dim, 1)
      dx = dout @ params['W'].T
      dW = x.T @ dout
      db = np.sum(dout, axis=0).T  
      grads = {'W': dW, 'b':db}
      
      return dx, grads

    return {'params': params, 'forward': forward, 'backward': backward}

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # if kind != 'relu':
    #   raise ValueError(f'{kind} activation not supported.')

    def forward(x):
      y=np.maximum(0,x)
      cache=x
      return y,cache

    def backward(dout, cache):
      x = cache
      #derivate of relu: 1 if x, else 0
      # dL/dx=dL/dy * dy/dx --> dx=dout*(x>0)
      dx=dout*(x>0)  #dx: shape(batch, in_dim)
      
      return dx, {}

    return {'params': {}, 'forward': forward, 'backward': backward}

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    var_W = np.sqrt(2/in_dim)
    mean_W = 0
    W = np.random.normal(mean_W, var_W, size=(in_dim, out_dim))
    b = np.zeros(out_dim)
    return (W, b)

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """

    '''
      using softmax activation for the output layer, we get the following:
      say 'y' is the final output, logits. we convert logits to prob (by satisfying the
      axioms of probability ofcourse)
      probs = softmax(y) --> probs = e^(y)/ sum(e^y), where
      probs is the multi-class predictions for each input vector.
    '''

    def loss_fn(logits, labels):
      # logits: shape(batch, C), C = raw class score(outs from the output layer) before softmax activation
      # labels: shape(batch,) int array of class indices in [0,C] i,e true labels for each input vector
      logits = logits - np.max(logits, axis=1, keepdims=True) # for numerical stability
      exp_logits = np.exp(logits)
      probs = exp_logits/np.sum(exp_logits, axis=1, keepdims=True)
      true_probs = probs[np.arange(logits.shape[0]), labels] # updates probs by taking only the probs for the true labels
      # true_probs --> cross-entropy loss L = -log(true_probs)
      # cross-entropy loss
      loss = -np.mean(np.log(true_probs)) # loss = L
      
      '''d_logits 
        = dL/d_logits = dL/d_true_probs * d_true_probs/d_probs * d_probs/d_exp_logits * d_exp_logits/d_logits
        = probs - one-hot_for_true_label (proof in copy after simplications)
      '''
      d_logits = probs.copy() # separate variable, we need probs if not .copy() probs get changed
      one_hot_y = np.zeros_like(logits)
      one_hot_y[np.arange(len(logits)), labels] = 1
      # one_hot_y[np.arange(len(logits)), labels] = 1
      d_logits = (probs - one_hot_y)/len(logits)
      return loss, d_logits 
    
    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    def forward(x):
      # forward as: input layer -> hidden layer 1 + activation -> ... -> outputs + loss
      caches = [] # needed during backprop
      for layer in layers:
        x, cache = layer['forward'](x)
        caches.append(cache)
      return x, caches
    
    def backward(dout, caches):
      grads_for_each_layer = [None]*len(layers) # grad=None for params-free layer eg. relu
      # perform backprop from the nth layer (the last layer)
      for i in reversed(range(len(layers))):
        dout, grads = layers[i]['backward'](dout, caches[i])
        grads_for_each_layer[i] = grads
      return dout, grads_for_each_layer
      

    return {
      "forward": forward,
      "backward": backward,
      "params": [layer['params'] for layer in layers]
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """

    # forward yields the following:
    logits, caches = model['forward'](x)
    ''' 
    loss_fn yields:
    d_logits = dL/d_logits where logits are inputs to softmax 
    '''
    loss, d_logits = loss_fn(logits, y)   

    # now that we obtained loss and d_logits as a param, we do backward
    dout, grads_for_each_layer = model['backward'](d_logits, caches)


    return loss, grads_for_each_layer

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement


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
    if kind != 'relu':
      raise ValueError(f'{kind} activation not supported.')

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

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

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


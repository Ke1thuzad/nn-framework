import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += self._handle_broadcasting(self.data, out.grad)
            other.grad += self._handle_broadcasting(other.data, out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += self._handle_broadcasting(self.data, other.data * out.grad)
            other.grad += self._handle_broadcasting(other.data, self.data * out.grad)
        out._backward = _backward
        return out

    def __pow__(self, n):
        out = Tensor(self.data**n, (self,), f'**{n}')
        def _backward():
            self.grad += (n * self.data**(n-1)) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward(): self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-15), (self,), 'log')
        def _backward(): self.grad += (1.0 / (self.data + 1e-15)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')
        def _backward(): self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            self.grad += np.broadcast_to(out.grad, self.data.shape)
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        res = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(res, (self,), 'max')
        def _backward():
            mask = (self.data == res)
            self.grad += mask * out.grad
        out._backward = _backward
        return out

    def _handle_broadcasting(self, target_data, grad):
        res = grad
        while res.ndim > target_data.ndim: res = res.sum(axis=0)
        for axis, size in enumerate(target_data.shape):
            if size == 1: res = res.sum(axis=axis, keepdims=True)
        return res

    def backward(self):
        topo = []; visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v); [build_topo(child) for child in v._prev]; topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

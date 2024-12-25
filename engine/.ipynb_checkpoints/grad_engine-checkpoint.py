import math

class GradNode:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label
        self.grad = 0.0
        self.require_grads = True

    def __repr__(self):
        return f"GradNode(data = {self.data})"
        
    def __add__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __radd__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
        
        
    def __sub__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0
        out._backward = _backward
        return out

    def __rsub__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(-self.data + other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad * -1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(other.data / self.data, (self, other), '/')
        def _backward():
            self.grad += out.grad * (-other.data/(self.data ** 2))
            other.grad += out.grad * (1/self.data)
        out._backward = _backward
        return out

    def __floordiv__(self, other):
        if type(other) != GradNode:
            return GradNode(self.data // other, (self, GradNode(other)), '//')
        return GradNode(self.data // other.data, (self, other), '//')
        
    def __pow__(self, other):
        if type(other) != GradNode:
            other = GradNode(other)
        out = GradNode(self.data ** other.data, (self, other), '**')
        def _backward():
            self.grad += out.grad * other.data * (self.data ** (other.data - 1)) 
            other.grad += out.grad * (self.data ** (other.data)) * math.log(self.data)
        out._backward = _backward
        return out

    def __abs__(self):
        return GradNode(abs(self.data), (self,), 'abs')

    def relu(self):
        out = GradNode(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        out = GradNode(self.data if self.data > 0 else alpha * self.data, (self,), 'LeakyReLU')
        def _backward():
            self.grad += (self.data > 0) * out.grad + (self.data <= 0) * alpha * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = GradNode(1 / (1 + math.exp(-self.data)), (self,), 'Sigmoid')
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = GradNode(math.tanh(self.data), (self,), 'Tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def prelu(self, alpha=0.01):
        out = GradNode(self.data if self.data > 0 else alpha * self.data, (self,), 'PReLU')
        def _backward():
            self.grad += (self.data > 0) * out.grad + (self.data <= 0) * alpha * out.grad
        out._backward = _backward
        return out

    def elu(self, alpha=1.0):
        out = GradNode(self.data if self.data > 0 else alpha * (math.exp(self.data) - 1), (self,), 'ELU')
        def _backward():
            self.grad += (self.data > 0) * out.grad + (self.data <= 0) * alpha * math.exp(self.data) * out.grad
        out._backward = _backward
        return out

    def linear(self):
        out = GradNode(self.data, (self,), 'Linear')
        def _backward():
            self.grad += out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        if self.require_grads == True:
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            # the first or starting node for the backprop chain has to have its derivative wrt to itself = 1.
            self.grad = 1
            for v in reversed(topo):
                v._backward()
            

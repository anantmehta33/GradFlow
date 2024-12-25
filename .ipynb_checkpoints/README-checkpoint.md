# GradFlow
Flow of gradients through the computational graph!!!


### \_\_add\_\_():
This is known as magic or dunder functions in python where we can redefine inbuilt python function. <br>
For example:<br>
result = obj1 + obj2 <br>
Translates to :<br>
result = obj1.\_\_add\_\_(obj2)

### Same for other operations also

### Convention: A single leading underscore (_) indicates that a parameter (or variable) is intended for internal use only and should not be accessed directly from outside the class or module.

### Convention: Use a trailing underscore (var_) rather than a leading underscore to avoid conflicts with Python keywords.

### Concept of Backprop:
#### First in forward pass all the initial node values are set. <br>
#### We start the backprop from the last node of the graph -- let this be final node. Now, final.grad() : $\frac{d}{dfinal}final = 1$ <br>
#### Now, at a node $c$, where $e = c + b$. We need c.grad() will be $\frac{d}{dc}final = \frac{d}{de}final * \frac{d}{dc}e = e.grad() * 1.0$ <br>
#### If, at a node $c$, where $e = c * b$. We need c.grad() will be $\frac{d}{dc}final = \frac{d}{de}final * \frac{d}{dc}e = e.grad() * b$ <br>
#### So, essentially for a child.gradient() = parent.gradient() * $\frac{d}{dchild}parent$ <br>
#### parent._backward() is storing the function to calculate the gradients of children wrt itself.
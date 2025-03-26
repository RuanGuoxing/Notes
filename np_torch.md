# Torch and numpy

## data manipulation


### stack

Concatenates a sequence of tensors along a new dimension.
same

```py

torch.stack(tensors, dim=0, *, out=None)
numpy.stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')

```

### concat (alias: concatenate in numpy, cat in torch)

Concat all arrays in the given dimension. All the arrays must have same size except the concat dimension.
The 2 functions get same results.

```py

numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
torch.cat(tensors, dim=0, *, out=None) 

```

Note: when axis is None in numpy, arrays are flattened before use.


### split

##### same

If the second argument is integer, they get same result along specified axis(dimension): 
split the array to equally chunk size sub-array. 

##### difference

- np will raise an exception if the split is not possible,
torch will get a smaller chunk of array in last if the size along the given dimension is not divisible by the integer.

- If the sections are given(like [2,4,8]), numpy will get [0:2], [2:4], [4:8], [8:]; torch get [0:2], [2:2+4], [2+4:2+4+8]


```py

torch.split(tensor, split_size_or_sections, dim=0)

numpy.split(ary, indices_or_sections, axis=0)


```

### repeat, np.tile, expand

torch.repeat unlike torch.expand(), this function copies the tensor’s data.
np.tile works like torch.repeat

```py
# If A.dim < len(reps), like shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication
# If A.ndim > d. Thus for an A of shape (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2)
numpy.tile(A, reps)        
torch.Tensor.repeat(*repeats) # Repeats this tensor along the specified dimensions.

>>> x = torch.tensor([1, 2, 3])
>>> x.repeat(4, 2)
tensor([[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]])
>>> x.repeat(4, 2, 1).size()
torch.Size([4, 2, 3])

>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])


```

### transpose

```py

numpy.transpose(a, axes=None) # If axes is None, default reverse of the order of axes.
numpy.ndarray.T               # same as transpose
numpy.permute_dims(a, axes=None)        # same sa transpose

torch.t(input)                  # input to be <= 2-D tensor
torch.transpose(input, dim0, dim1)  # same sa .t, using permute if dims > 2.
torch.permute(input, dims)      

>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> torch.permute(x, (2, 0, 1)).size()
torch.Size([5, 2, 3])
```


## bitwise operation

### bitwise_and

same.

Compute the bit-wise AND of two arrays element-wise.

```py
# The number 13 is represented by 00001101. Likewise, 17 is represented by 00010001. 
# The bit-wise AND of 13 and 17 is therefore 000000001, or 1:
np.bitwise_and(13, 17)
1

x1 = np.array([2, 5, 255])
x2 = np.array([3, 14, 16])
x1 & x2
array([ 2,  4, 16])

```

## index, sort

### nonzero

get the index of nonzero element. as_tuple is True in numpy default.

```py

torch.nonzero(input, *, out=None, as_tuple=False)
torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
tensor([[ 0],
        [ 1],
        [ 2],
        [ 4]])
torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
...                             [0.0, 0.4, 0.0, 0.0],
...                             [0.0, 0.0, 1.2, 0.0],
...                             [0.0, 0.0, 0.0,-0.4]]))
tensor([[ 0,  0],
        [ 1,  1],
        [ 2,  2],
        [ 3,  3]])
torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
(tensor([0, 1, 2, 4]),)
torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
...                             [0.0, 0.4, 0.0, 0.0],
...                             [0.0, 0.0, 1.2, 0.0],
...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
(tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
torch.nonzero(torch.tensor(5), as_tuple=True)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a > 3
array([[False, False, False],
       [ True,  True,  True],
       [ True,  True,  True]])
np.nonzero(a > 3)
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
(a > 3).nonzero()
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
```

### where

same

```py

torch.where(condition, input, other, *, out=None)       # The tensors condition, input, other must be broadcastable.(scalar will be fine). same as numpy

x
tensor([[-0.4620,  0.3139],
        [ 0.3898, -0.7197],
        [ 0.0478, -0.1657]])
>>> torch.where(x > 0, 1.0, 0.0)
tensor([[0., 1.],
        [1., 0.],
        [1., 0.]])


array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.where(a < 5, a, 10*a)
array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

np.where([[True, False], [True, True]],
         [[1, 2], [3, 4]],
         [[9, 8], [7, 6]])
array([[1, 8],
       [3, 4]])

```



### sort, argsort

numpy return a sorted array but torch get a sorted array and indices.


```py

numpy.sort(a, axis=-1, kind=None, order=None, *, stable=None)
torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)
torch.argsort(input, dim=-1, descending=False, stable=False)            # second value returned by torch.sort()
numpy.argsort(a, axis=-1, kind=None, order=None, *, stable=None)        # just like above

a = np.array([[1,4],[3,1]])
np.sort(a)                # sort along the last axis if axis is not given
array([[1, 4],
       [1, 3]])
np.sort(a, axis=None)     # sort the flattened array
array([1, 1, 3, 4])


x = torch.randn(3, 4)
>>> sorted, indices = torch.sort(x)
>>> sorted
tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
        [-0.5793,  0.0061,  0.6058,  0.9497],
        [-0.5071,  0.3343,  0.9553,  1.0960]])
>>> indices
tensor([[ 1,  0,  2,  3],
        [ 3,  1,  0,  2],
        [ 0,  3,  1,  2]])

```


### argmax

same. defaults are keepdim=False.

```py

numpy.argmax(a, axis=None, out=None, *, keepdims=<no value>)
torch.argmax(input, dim, keepdim=False)

a = torch.randn(4, 4)
>>> a
tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [-0.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])
>>> torch.argmax(a, dim=1)
tensor([ 0,  2,  0,  1])


```

### mean

```py

# axis could be tuple, it will compute the average along multi-axes.
numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)
torch.mean(input, dim, keepdim=False, *, dtype=None, out=None)

a = np.array([[1, 2], [3, 4]])
np.mean(a)
2.5
np.mean(a, axis=0)
array([2., 3.])
np.mean(a, axis=1)
array([1.5, 3.5])

```


### rand, randn, random


```py

numpy.random.rand(d0, d1, ..., dn)        # Random values in a given shape. random samples from a uniform distribution over [0, 1)
torch.rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)

# Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution)
torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
numpy.random.randn(d0, d1, ..., dn)

# Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive)
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
numpy.random.randint(low, high=None, size=None, dtype=int)


np.random.seed(0)
torch.seed(0)
```

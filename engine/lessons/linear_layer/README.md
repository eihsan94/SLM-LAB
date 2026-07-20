# Linear layer

The linear layer computes:

```text
Y = X * W^T + b
```

For one sample `x` and one neuron `n`:

```text
y[n] = x[0] * W[n,0]
     + x[1] * W[n,1]
     + ...
     + b[n]
```

The complete algorithm and visualization connection are intentionally together
in [lesson.cpp](lesson.cpp). Read the three nested loops first, then follow the
values into the short `show::*` calls at the bottom.

When this algorithm is understood and another lesson needs it, promote
`linear_forward()` into `libs/math/linear.hpp`. That is how the reusable math
library grows throughout the syllabus.

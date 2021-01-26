# pytransfer

Library for implementing the transfer operator approach.

### The approach

Consider a discrete (or discretized) dynamical system:
```math
x(t+\Delta t) = f(x(t)),
```
where $`x\in\mathbb R^n`$ and $`f`$ is the iterated map $`f:\mathbb R^n \to \mathbb R^n`$.

Then, the transfer operator $`\mathcal T`$ is the corresponding map over probability measures, $`u\in\mathcal M`$:
```math
u(t+\Delta t) = \mathcal T u(t)
```

We get to exchange possibly horribly non-linear intractable (but finite dimensional) dynamics for well-behaving linear (but infinite dimensional) dynamics.

But we can usually get by with a finite rank approximation, which we can compute directly from data.

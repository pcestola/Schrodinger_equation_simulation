# 1D Time Dependent Schrödinger Equation Solver
This project introduces a solver for the 1D Schrödinger equation, leveraging the computational power of GPUs and the Finite Difference Method (FDM).
The Schrödinger equation, foundational in quantum mechanics since its proposal by Erwin Schrödinger in 1925, describes the behavior of quantum particles.
However, its complex nature often necessitates numerical methods for solution. The core of our simulation is governed by the time-dependent Schrödinger equation in one dimension:
$$i\hbar\frac{\partial}{\partial t}\Psi(x,t) = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}\Psi(x,t) + V(x)\Psi(x,t)$$
where $i$ is the imaginary unit, $\hbar$ is the reduced Planck's constant, $m$ is the mass of the particle, $\Psi(x,t)$ is the wave function of the particle at position $x$ and time $t$ and $V(x)$ is the potential energy as a function of position.

This project begins by setting an initial condition and potential for the quantum system under consideration. Dirichlet boundary conditions are applied at the edges of the domain. The initial condition describes the initial state of the quantum particle, while the potential dictates the external forces acting on the particle.
As the simulation concludes, the evolution of the quantum system's wave function under the specified initial conditions and potential is represented as a 2-dimensional numpy ndarray. This representation can then compiled into a GIF using __animation.py__.
The resulting GIF captures the behavior over time, offering educational insights into the simulated scenario.

## 💻Code
All the code is written using __python 3.10.12__. Below there is an example of the GIF that can be produced with the code. The code to produce this GIF is available in __test.ipynb__.

<p align="center">
  <img src="/output_gifs/schrodinger.gif" width="600" />
</p>

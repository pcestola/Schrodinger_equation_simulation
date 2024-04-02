# 1D Time-Dependent Schrödinger Equation Solver
This project introduces a solver for the 1D Schrödinger equation using the Finite Difference Method (FDM). The code allows the calculation to be performed on the GPU.
The Schrödinger equation, foundational in quantum mechanics since its proposal by Erwin Schrödinger in 1925, describes the behavior of quantum particles.
However, its complex nature often necessitates the use of numerical methods for solutions. The core of the simulation is governed by the time-dependent Schrödinger equation in one dimension:
$$i\hbar\frac{\partial}{\partial t}\Psi(x,t) = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}\Psi(x,t) + V(x)\Psi(x,t)$$
where $i$ is the imaginary unit, $\hbar$ is the reduced Planck's constant, $m$ is the mass of the particle, $\Psi(x,t)$ is the wave function of the particle at position $x$ and time $t$ and $V(x)$ is the potential energy as a function of position.

This project begins by setting an initial condition and potential for the quantum system under consideration. Dirichlet boundary conditions are applied at the edges of the domain.
As the simulation concludes, the evolution of the quantum system's wave function under the specified initial conditions and potential is represented as a 2-dimensional numpy ndarray. This representation can then be compiled into a GIF using **animation.py**.
The resulting GIF captures the behavior over time, offering educational insights into the simulated scenario.

## 💻Code
All the code is written using **Python 3.10.12**. Below there are some examples of the GIFs that can be produced with the code. The code to produce the first GIF is available in **test.ipynb**.

### Tunneling
<p align="center">
  <img src="/output_gifs/schrodinger.gif" width="600" />
</p>

### Harmonic oscillator
<p align="center">
  <img src="/output_gifs/armonic_oscillator.gif" width="600" />
</p>

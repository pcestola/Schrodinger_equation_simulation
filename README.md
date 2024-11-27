# 1D Time-Dependent Schrödinger Equation Solver

This project introduces a solver for the 1D Schrödinger equation using the Finite Difference Method (FDM). The code leverages GPU acceleration for efficient computations, enabling the exploration of quantum mechanics simulations in a practical and educational way.

The Schrödinger equation, proposed by Erwin Schrödinger in 1925, is foundational in quantum mechanics and describes the behavior of quantum particles. Due to its mathematical complexity, numerical methods are required for solving it. This solver specifically addresses the **time-dependent Schrödinger equation in one dimension**, expressed as:

$$i\hbar\frac{\partial}{\partial t}\Psi(x,t) = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}\Psi(x,t) + V(x)\Psi(x,t)$$  

where:  
- $i$ is the imaginary unit,  
- $\hbar$ is the reduced Planck constant,  
- $m$ is the particle's mass,  
- $\Psi(x,t)$ is the wave function at position $x$ and time $t$,  
- $V(x)$ is the potential energy as a function of position.  

## 🛠️ Project Features
1. **Initial Conditions and Potentials**:  
   The simulation begins with user-defined initial conditions and potential for the quantum system.
2. **Boundary Conditions**:  
   Dirichlet boundary conditions are applied at the edges of the domain to ensure well-defined solutions.
3. **Wave Function Evolution**:  
   The output is represented as a 2D `numpy.ndarray`, showing the evolution of the wave function over time.
4. **Visualization**:  
   Results can be compiled into an animated GIF using **animation.py**, providing visual insights into the dynamics of the quantum system.

---

## 💻 Code
All code is written in **Python 3.10.12** and supports GPU acceleration. Below are examples of GIFs generated using this solver. The code to reproduce the first example is available in **test.ipynb**.

---

### 🔍 Examples

#### **Tunneling**
This example shows quantum tunneling, where a particle traverses a potential barrier that it classically could not surpass.

<p align="center">
  <img src="/output_gifs/Tunneling.gif" width="600" />
</p>

#### **Harmonic Oscillator**
A quantum harmonic oscillator example, showing the oscillatory behavior of a particle in a quadratic potential well.

<p align="center">
  <img src="/output_gifs/Harmonic_oscillator.gif" width="600" />
</p>

#### **Coming Soon**
A teaser for additional simulations to be included in future updates.

<p align="center">
  <img src="/output_gifs/schrodinger.gif" width="600" />
</p>

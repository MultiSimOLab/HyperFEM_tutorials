
<!-- # HyperFEM :construction: :construction: :construction: **Work in progress** :construction: :construction: :construction: -->

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmartfrut.github.io/HyperFEM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmartfrut.github.io/HyperFEM.jl/dev/)
[![Build Status](https://github.com/MultiSimOLab/HyperFEM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MultiSimOLab/HyperFEM/actions/workflows/ci.yml?branch=main)
[![Coverage](https://codecov.io/gh/jmartfrut/HyperFEM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmartfrut/HyperFEM.jl)

# HyperFEM tutorials <img src="https://github.com/jmartfrut/HyperFEM/blob/main/docs/imgs/logo.png?raw=true"  width="40" title="HyperFEM logo">

<div align="justify" style="margin-left: 40px; margin-right: 40px;">

This repository contains a set of tutorials designed to help users learn how to simulate hyperelastic materials and solve multiphysic problems using the Finite Element Method (FEM) in Julia with [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl).

HyperFEM.jl is built on top of the [Gridap.jl](https://github.com/gridap/Gridap.jl) ecosystem, providing specialized tools for multiphysics hyperelastic simulations. The initial tutorials demonstrate the core usage of HyperFEM, and these are recommended for new users.
 

</div>

## Installation
1- Clone the repository:
git clone https://github.com/MultiSimOLab/HyperFEM_tutorials.git
```
cd HyperFEM_tutorials
```
2- Open the Julia REPL, type `]` to enter package mode, and activate de environment:
```julia
pkg> activate .
```

3- Install dependencies:
```julia
pkg> instantiate
```

## Examples
The HyperFEM tutorials include a wide range of examples, carefully selected to demonstrate the toolbox's capabilities. Each example focuses on a specific type of problem, from basic PDEs to complex multiphysics and optimization scenarios. These examples are ideal for understanding both the theoretical formulation and the practical implementation of FEM simulations in Julia.

```julia
Example 1: Poisson
# Introduces fundamental FEM concepts and demonstrates solving a simple Poisson equation.

Example 2: Hyperelastic beam stretching
# Illustrates large deformation analysis of a hyperelastic beam, showcasing material nonlinearity.

Example 3: Hyperelastic cylinder (4 fibres model) under internal pressure
# Demonstrates anisotropic hyperelastic modeling with fiber-reinforced materials under internal loading.

Example 4: Electromechanical beam
# Introduces coupled electromechanical simulations, highlighting interactions between mechanical and electrical fields.

Example 5: Anisotropic Electromechanical beam
# Shows the effect of anisotropic material behavior in coupled electromechanical problems.

Example 6: Hyperelastic contact with third medium
# Covers contact mechanics involving hyperelastic materials interacting with a third body.

Example 7 Topology optimization of hyperelastic cantilever
# Demonstrates optimization techniques applied to hyperelastic structures for design improvement.

Example 8: Magnetomechanical beam
# Illustrates magnetomechanical coupling simulations, integrating magnetic and mechanical field interactions.
```


## How to cite HyperFEM

In order to give credit to the HyperFEM contributors, we ask that you please reference the paper:

C. Perez‐Garcia, R. Ortigosa, J. Martínez‐Frutos, and D. Garcia‐Gonzalez, **Topology and material optimization in ultra-soft magnetoactive structures: making advantage of residual anisotropies.** Adv. Mater. (2025): e18489. https://https://doi.org/10.1002/adma.202518489
 

along with the required citations for [Gridap](https://github.com/gridap/Gridap.jl).


# Project funded by:
 
- Grants PID2022-141957OA-C22/PID2022-141957OB-C22  funded by MCIN/AEI/ 10.13039/501100011033  and by ''ERDF A way of making Europe''


 <p align="center"> 
&nbsp; &nbsp; &nbsp; &nbsp;
<img alt="Dark"
src="https://github.com/MultiSimOLab/HyperFEM/blob/main/docs/imgs/aei.png?raw=true" width="70%">
</p>
 
#  Contact

Contact the project administrator [Jesús Martínez-Frutos](jesus.martinez@upct.es) for further questions about licenses and terms of use.
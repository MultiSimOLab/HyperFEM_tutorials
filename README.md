
<!-- # HyperFEM :construction: :construction: :construction: **Work in progress** :construction: :construction: :construction: -->

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmartfrut.github.io/HyperFEM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmartfrut.github.io/HyperFEM.jl/dev/)
[![Build Status](https://github.com/MultiSimOLab/HyperFEM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MultiSimOLab/HyperFEM/actions/workflows/ci.yml?branch=main)
[![Coverage](https://codecov.io/gh/jmartfrut/HyperFEM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmartfrut/HyperFEM.jl)

# HyperFEM tutorials <img src="https://github.com/jmartfrut/HyperFEM/blob/main/docs/imgs/logo.png?raw=true"  width="40" title="HyperFEM logo">

<div align="justify" style="margin-left: 40px; margin-right: 40px;">

This repository contains a set of tutorials designed to help users learn how to simulate hyperelastic materials and solve multiphysic problems using the Finite Element Method (FEM) in Julia with the [HyperFEM](https://github.com/gridap/Gridap.jl) toolbox.

HyperFEM is built on top of the Gridap.jl ecosystem, providing specialized tools for multiphysics hyperelastic simulations. The initial tutorials demonstrate the core usage of HyperFEM, and these are recommended for new users.
 

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

## Installation


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
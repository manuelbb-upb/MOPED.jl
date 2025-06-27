# MOPED
## **M**ulti-**O**bjective **P**roblem and **E**xperimentation **D**ata

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/MOPED.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/MOPED.jl/dev/)
[![Build Status](https://github.com/manuelbb-upb/MOPED.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/manuelbb-upb/MOPED.jl/actions/workflows/CI.yml?query=branch%3Amain)

![Moped Logo](./tex/logo.png)

## Disclaimer
This is my personal research repository for benchmarking multi-objective algorithms.

Please don't consider anything *stable* you see here.
Interfaces might change, and parts of this package could be split out.

## Goals
1) I hope to develop a common interface for many (nonlinear) multi-objective test problems.
2) Curate a library of test problems.  
   Problems can be unconstrained, have variable constraints, linear inequality or equality constraints, 
   or nonlinear inequality or equality constraints.
3) Facilitate interoperability between different solvers and different packages.
4) Provide ways to compute common performance indicators.

### Current Status

I have spend way to much time thinking about 1) + 3), so there are no problems here, yet.
But I think, the bridging mechanism, that has been developed is really really neat!

## Related

* [NonlinearCG package](https://github.com/manuelbb-upb/NonlinearCGCode): A repository with  
  some nonlinear CG algorithms and MOO test problems.
* [This repository](https://github.com/manuelbb-upb/MOBenchmarks) has code to wrap the  
  Fortran test problem library [TESTMO](https://github.com/DerivativeFreeLibrary/TESTMO). 
  We compared [`Compromise.jl`](https://github.com/manuelbb-upb/Compromise.jl/tree/main) against
  [`DFMO`](https://github.com/DerivativeFreeLibrary/DFMO).  
  There were a lot of awkward code manipulations.  
  [This wrapper](https://github.com/manuelbb-upb/DFMOWrapper.jl) is much nicer!
* [`pymoo`](https://github.com/anyoptimization/pymoo) is a Python package with test problems  
  and algorithms.
* [`MathOptInterface.jl`](https://github.com/jump-dev/MathOptInterface.jl) provided most of  
  the important ideas concerning **bridging**.
* [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl)  
  inspired and influenced the caching mechanism. We support automatic differentiation.
* [`MultiObjectiveAlgorithms.jl`](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl),  
  a collection of `MOI`-backed algorithms.
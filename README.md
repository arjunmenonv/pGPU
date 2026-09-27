## Contribution Guidelines

1. This is meant to be a private fork for building out the simulator infra and test-cases. We are likely to end up having code in the commit history here that leaks the quirks in the simulator. 
2. The idea is to have a clean `main` branch that has just the details required for working on the problem, and have the `main` branch sync'd to public fork of this repository.
3. Please make sure you commit all your changes to a private repo following the convention: `<name>/<branch_name>`. For changes that need to be sync'd to the `main` branch, raise a PR. 
4. Things that are NOT planned to be exposed to `main` or end-user: correct driver code for XOR swizzling, kernel implementation for softmax, pytest routines for arch features, arch bug fixes, detailed docs of the arch features, etc.

# pGPU Simulator 

`pGPU` is a simulator for a minimal GPU-like device, entirely in Python, with its own mini-ISA, DRAM, Scratch, Register-File and SM models. 

## Problem Description

The target problem is to implement a kernel in pGPU's software framework to compute the attention map between two matrices:

$$\text{Output} = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right)$$

<More details to follow>

## Setup Instructions

## Simulator Docs

Watch this space!

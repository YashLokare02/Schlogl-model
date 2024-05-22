# Schlogl Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?logo=Qiskit&logoColor=white)
![Static Badge](https://img.shields.io/badge/paddle-quantum?style=for-the-badge&logoColor=%23FF0000&label=PaddleQuantum&link=https%3A%2F%2Fgithub.com%2FPaddlePaddle%2FQuantum)

This repository contains code to implement the classical and quantum subroutines for the paper ***Modeling Stochastic Chemical Kinetics on Quantum Computers***.

Link to the paper: [arXiv:2404.08770](https://arxiv.org/abs/2404.08770).

**Authors**: T. Kabengele, Y. M. Lokare, J. B. Marston, and B. M. Rubenstein. 

The following Qiskit software versions are needed to run **QPE + VQSVD**: 
- qiskit-terra: 0.24.0
- qiskit-aer: 0.12.0
- qiskit-ibmq-provider: 0.20.2
- qiskit: 0.43.0

The following Qiskit software versions are needed to run **VQD**: 
- qiskit-terra: 0.25.1
- qiskit-algorithms: 0.1.0
- qiskit: 0.44.1

Version of Paddle Quantum required to run numerical simulations: **2.4.0**; refer [PaddleQuantum](https://github.com/PaddlePaddle/Quantum) for more details. 

We use the Python ***poetry*** package management system to manage all package dependencies in Python. 

**Note**: The DD analysis and/or results contained in this repository are irrelevant for the present study. 

## Citation

Cite this paper as follows: 

```
@misc{kabengele2024modeling,
      title={Modeling Stochastic Chemical Kinetics on Quantum Computers}, 
      author={Tilas Kabengele and Yash M. Lokare and J. B. Marston and Brenda M. Rubenstein},
      year={2024},
      eprint={2404.08770},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

# cortecs
[![Tests](https://github.com/arjunsavel/cortecs/actions/workflows/python-package.yml/badge.svg)](https://github.com/arjunsavel/cortecs/actions/workflows/python-package.yml)
[![Maintainability](https://api.codeclimate.com/v1/badges/4eb53795313af153f4cd/maintainability)](https://codeclimate.com/github/arjunsavel/cortecs/maintainability)

A Python package for decreasing the memory footprint of opacity functions. The primary functionality is compressing opacity functions with varying flexibility. Current methods include
- polynomial fitting
- PCA-based fitting
- neural network fitting


All fits are currently made in along the temperature and pressure axes. 

Additionally, `cortecs` can chunk up opacity functions. The radiative transfer problem can often be cast as embarassingly parallel, so each chunk can be sent to a different CPU.

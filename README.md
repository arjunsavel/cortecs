# cortecs
[![Tests](https://github.com/arjunsavel/cortecs/actions/workflows/python-package.yml/badge.svg)](https://github.com/arjunsavel/cortecs/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/arjunsavel/cortecs/graph/badge.svg?token=90S3STBO5X)](https://codecov.io/gh/arjunsavel/cortecs)
[![Maintainability](https://api.codeclimate.com/v1/badges/4eb53795313af153f4cd/maintainability)](https://codeclimate.com/github/arjunsavel/cortecs/maintainability)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CodeQL](https://github.com/arjunsavel/cortecs/actions/workflows/codeql.yml/badge.svg)](https://github.com/arjunsavel/cortecs/actions/workflows/codeql.yml)
[![Documentation Status](https://readthedocs.org/projects/cortecs/badge/?version=latest)](https://cortecs.readthedocs.io/en/latest/?badge=latest)
[![Paper compilation](https://github.com/arjunsavel/cortecs/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/arjunsavel/cortecs/actions/workflows/draft-pdf.yml)


A Python package for decreasing the memory footprint of opacity functions. The primary functionality is compressing opacity functions with varying flexibility. Current methods include
- polynomial fitting
- PCA-based fitting
- neural network fitting


All fits are currently made in along the temperature and pressure axes. 

Additionally, `cortecs` can chunk up opacity functions. The radiative transfer problem can often be cast as embarassingly parallel, so each chunk can be sent to a different CPU.

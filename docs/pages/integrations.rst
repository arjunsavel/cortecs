Integrations
----------------
:code:`cortecs` uses the below freely available integrations and services:

Code style
------------
- `black <https://black.readthedocs.io/en/stable/>`_: a Python code-formatter.
  It generally conforms to `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_
  standards, but it does so in a very deterministic manner. The pre-commit configuration is located in :code:`.pre-commit-config.yaml`.

- `isort <https://isort.readthedocs.io/en/latest/>`_: a tool for sorting imports
  in Python files. Similarly to `black`, the basic pre-commit configuration is located
  in :code:`.pre-commit-config.yaml`.

- `yamllint <https://github.com/adrienverge/yamllint>`_: lints our YAML files,
  checking them for bugs, syntax errors, and general style. The associated
  configuration file is :code:`.yamllint`, which attaches to the pre-commit hook.

Continuous integration
-----------------------

- `Codecov <https://codecov.io/gh>`_: checks what percentage of our code base
  is covered in our automated tests. Ideally, we'd like to keep our coverage above
  95% if it is in a production environment. Relevant configurations are noted in
  :code:`codecov.yml`.

- `GitHub Actions <https://github.com/features/actions>`_: A number of the
  services described in this page are implemented as GitHub Actions, which manifest as
  `checks <https://developer.github.com/v3/checks/>`_ on different commits. In
  general, we would like all of our checks to pass before pull requests are
  merged. Configuration files for GitHub Actions are contained in
  :code:`.github/workflows`.

- `pre-commit hooks <https://pre-commit.com/>`_: these can be applied to any
  local commits. The associated configuration file is :code:`.pre-commit-config.yml`.

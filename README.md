# Cost optimization in frequency control of unbalanced distribution systems

## Prerequisites
- `PyEnv`: Ensures correct python version (as per `.python-version`)
- `PipEnv`: Ensures dependencies & locks down package versions

## Running Code
1. Ensure you are within the _virtualenv_: `pipenv shell`


## Troubleshooting

### Jupyter Notebook
**Problem:** Jupyter Notebook fails to run. Usually indicating an issue of invalid syntax pointing to typing hints.
**Fix:** Ensure the python kernel running in jupyter is of the correct python version (tested with `3.7.3`).
To show version in any notebook:
```Python
    from platform import python_version
    print(python_version())
```
To update the kernel in the jupyter installation to your currently installed python version (using pip)
```Bash
    python3 -m pip install ipykernel
    python3 -m ipykernel install --user
```

**Problem:** Modules installed through `pipenv` are not being found within a jupyter notebook
**Fix:** You need to ensure the kernel created points to the dedicated virtual environment in which `pipenv` installs the packages.
First you need to find out the name given to the `virtualenv` created by `pipenv`. In a terminal session, navigate to the project's root and type:
```Bash
  $ pipenv --venv
  => ~/.local/share/virtualenvs/VIRTUAL_ENV_NAME
```
With the virtual env name in hand, it is now time to declare a new kernel for jupyter pointing to that specific environment
```Bash
  pipenv run python3 -m ipykernel install --user --name=VIRTUAL_ENV_NAME
```
Finally, open the desired notebook and ensure the correct kernel is used using the top menu: Kernel => Change Kernel => VIRTUAL_ENV_NAME

**Alternate fix:** If nothing else works, as a final alternative one could install all the packages listed in the `pipfile` using whatever python package manager they prefer (`conda`, `pip`, etc). Ensuring the versions match the ones described in the `pipfile`. This is **not recommended**, as the preferred way would be to ensure the project is running as initially specified, but it can be an alternative where nothing else seems to work.

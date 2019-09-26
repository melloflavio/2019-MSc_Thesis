# Cost optimization in frequency control of unbalanced distribution systems

## Prerequisites
- `PyEnv`: Ensures correct python version (as per `.python-version`)
- `PipEnv`: Ensures dependencies & locks down package versions

## Running Code
1. Starting with a system with python v3.7.3 and the corresponding pip version installed
    **Note:** `Pyenv` and `pipenv` are optional, but facilitate the process of ensuring the correct versions of the libraries
   1. If using pipenv, ensure you are within the _virtualenv_: `pipenv shell` or issue commands with `pipenv run <command>`
2. Install all libraries specified in section B.2. Ensure the correct versions are being installed.
   1. If using `pipenv` this can be done running the command `pipenv install`, which automatically installs all the libraries as specified in the pipfile
   2. If relying on pip only, each library can be installed individually
3. Start the jupyter notebook server: (`jupyter notebook`)
4. Run the experiments found in the form of notebooks in the `./app/experiments` folder
5. A template notebook can also be found which provide the blueprint for running future experiments. `./app/experiments/Template-Experiment.ipynb`


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

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

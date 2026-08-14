# Installation

AdaHIsomap requires both **Python** and **Julia**.

The persistent-homology computation used by AdaHIsomap is implemented in Julia and is called from Python through **PyJulia**. Therefore, both Python and Julia must be installed and configured before running the program.

The commands below assume that you begin in the root directory of the cloned repository.

First, enter the implementation directory:

```bash
cd AdaMapper_AdaHIsomap
```

All remaining commands should be run from this directory unless otherwise stated.

---

## 1. Python Setup

The code was developed and tested with:

```text
Python 3.12.3
```

We recommend creating a virtual environment before installing the required Python packages.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required Python packages from `requirements.txt`:

### macOS / Linux

```bash
python3 -m pip install -r requirements.txt
```

### Windows

```bash
python -m pip install -r requirements.txt
```

The main Python dependencies include:

- NumPy
- SciPy
- scikit-learn
- Matplotlib
- NetworkX
- PyJulia

> **Note:** The interactive interface uses Tkinter and the Matplotlib `TkAgg` backend. Your Python installation must therefore include Tk support. Tkinter is not listed in `requirements.txt` because it is not normally installed as a standard `pip` package.

---

## 2. Julia Setup

Julia is required to compute the persistent-homology information used by the AdaMapper pipeline.

The code was developed using:

```text
Julia 1.11
```

Install Julia and make sure that the `julia` executable is available through your system `PATH`.

Verify the installation from a terminal:

```bash
julia --version
```

If this command displays the installed Julia version, Julia is accessible correctly from the command line.

---

## 3. Install the Required Julia Packages

The Julia persistent-homology implementation is located in:

```text
extract_loop_info.jl
```

It uses the following external Julia packages:

- `Distances`
- `Ripserer`
- `JSON`
- `PyCall`

Start Julia:

```bash
julia
```

Then install the required packages:

```julia
using Pkg

Pkg.add("PyCall")
Pkg.add("Distances")
Pkg.add("Ripserer")
Pkg.add("JSON")
```

After installation, exit Julia:

```julia
exit()
```

---

## 4. Configure the Python-Julia Connection

AdaHIsomap uses **PyJulia** to call Julia from Python.

The Python-Julia interface is initialized in:

```text
compute_H1_features.py
```

and this Python module loads:

```text
extract_loop_info.jl
```

After installing both the Python and Julia dependencies, start Python from the same virtual environment in which you installed `requirements.txt`.

### macOS / Linux

```bash
python3
```

### Windows

```bash
python
```

Then run:

```python
import julia
julia.install()
```

This configures Julia's `PyCall` package for the Python interpreter currently being used.

Exit Python:

```python
exit()
```

---

## 5. Verify the Python-Julia Connection

Before running AdaHIsomap, verify that Python can successfully communicate with Julia.

### macOS / Linux

```bash
python3 -c "from julia.api import Julia; jl = Julia(compiled_modules=False); from julia import Main; print(Main.eval('1+1'))"
```

### Windows

```bash
python -c "from julia.api import Julia; jl = Julia(compiled_modules=False); from julia import Main; print(Main.eval('1+1'))"
```

A successful configuration should print:

```text
2
```

---

## 6. Persistent-Homology Computation

The Python file

```text
compute_H1_features.py
```

initializes Julia through PyJulia and calls:

```text
extract_loop_info.jl
```

The Julia code uses `Ripserer.jl` to compute the one-dimensional persistent-homology (`H1`) features required by AdaMapper.

---

## 7. Run AdaHIsomap

Inside the `AdaMapper_AdaHIsomap` directory, run:

### macOS / Linux

```bash
python3 Run_AdaHIsomap.py
```

### Windows

```bash
python Run_AdaHIsomap.py
```

The program should be run from this directory because the current implementation uses relative paths such as:

```text
./data
./results
```

The interactive interface will then guide you through the experiment workflow.

---
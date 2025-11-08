# cs229

# Setting up
Need to set up [Flow](https://github.com/flow-project/flow) and [SUMO](https://www.eclipse.org/sumo/)
## Requirements

* macOS or Linux
* Python 3.8 (via `pyenv`)
* CMake, make, g++ (for building SUMO)

## Setup 

### 1. Clone the Repo and Submodules

```bash
git clone --recurse-submodules https://github.com/your-username/your-repo.git
cd your-repo
```

Or manually:

```bash
git clone https://github.com/flow-project/flow.git
cd flow
pip install -e .

cd ..
git clone https://github.com/eclipse/sumo.git
```

### 2. create & activate python env

```bash
pyenv install 3.8.18
pyenv virtualenv 3.8.18 flow_env
pyenv activate flow_env
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
cd flow
pip install -e .
```

### 4. Build and Configure SUMO

```bash
cd ../sumo
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Set environment variables:

```bash
export SUMO_HOME=$(pwd)/..
export PATH=$SUMO_HOME/bin:$PATH
```

Make this persistent:

```bash
echo 'export SUMO_HOME=$(pwd)/..' >> ~/.zshrc
echo 'export PATH=$SUMO_HOME/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
```

### 5. verify flow setup w/ unit tests

```bash
which sumo
sumo --version
python -m unittest discover flow
```

## Directory Structure

| Folder         | Description                              |
| -------------- | ---------------------------------------- |
| `flow/`        | Flow RL framework (installed via source) |
| `sumo/`        | SUMO simulator (installed via source)    |
| `experiments/` | Your custom code                         |

## .gitignore Recommendations

```
sumo/
flow/
build/
__pycache__/
*.pyc
```

If using git submodules, do **not** ignore `sumo/` or `flow/`.

### [IMPORTANT] Dependency Notes
building on the [Flow](https://github.com/flow-project/flow) traffic simulation library and modifies or pins several key dependencies to ensure compatibility with macOS + Python 3.8

Instead of modifying `flow/requirements.txt`, we define our environment in the top-level `requirements.txt` for reproducibility and clarity.

To install dependencies:

```bash
python -m venv flow_env
source flow_env/bin/activate
pip install -r requirements.txt

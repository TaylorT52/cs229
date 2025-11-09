# cs229

# Setting up
Need to set up [Flow](https://github.com/flow-project/flow) and [SUMO](https://www.eclipse.org/sumo/)
## Requirements

* macOS or Linux
* Python 3.8 (via `pyenv`)
* CMake, make, g++ (for building SUMO)
* [SUMO](https://github.com/eclipse/sumo) (built from source)
* [Flow](https://github.com/flow-project/flow) (installed from source)

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

- `ray==1.9.0` instead of older 0.8.x
- `redis>=3.5.0,<5.0.0` (to satisfy ray vs Flow requirements)
- `torch==1.10.0` (instead of 1.4.0 in original Flow)
- `gym==0.21.0`, `numpy==1.21.6`, `scipy==1.7.3` (newer, stable versions)

To install dependencies:

```bash
python -m venv flow_env
source flow_env/bin/activate
pip install -r requirements.txt

## Testing the Installation

Once you've built SUMO and installed Flow, you can run a simple built-in simulation to verify everything is working:

```bash
RENDER=drgb PYTHONPATH=$PWD python flow/examples/simulate.py ring
```

If everything is set up correctly, a GUI window should appear showing vehicles driving on a ring road. You should see output like:

```
Simulation started with time: 0.00
Simulation ended at time: 150.10
```

## GUI Rendering on macOS

If the SUMO GUI fails to launch due to a display error (`FXApp::openDisplay: unable to open display :0.0`), try:

```bash
export DISPLAY=:0
```

May also need to install [XQuartz](https://www.xquartz.org/) and log out / log back in.

## Activating Your Environment Permanently

Automatically activate your `flow_env` virtualenv on shell startup (assuming you're using `zsh` and `pyenv`):

Add this to your `~/.zshrc`:

```bash
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate flow_env
```

Then reload:

```bash
source ~/.zshrc
```
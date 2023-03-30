# TelescopeSim
Utilize HCIPy to simulate telescope optics, atmosphere for novel distirbuted aperture designs and AO concepts

## Install
```bash
# create virtual env and install dependencies
conda create -n dasie python=3.10 pip
pip install numpy matplotlib scipy hcipy
pip install pyrallis

# install telescope_sim package
git clone https://github.com/phelps-matthew/TelescopeSim.git
cd TelescopeSim
pip install -e .
```

## Usage
```bash
# navigate to .../TelescopeSim/telescope_sim

# inspect simulator CLI flags
python simlulate.py --help

# run simulation with defaults
python simulate.py

# run simulation based on yaml configuration
python simulate.py --config_path ./configs/default.yaml

# run simulation and override yaml with CLI flags
python simulate.py --config_path ./configs/default.yaml --num_steps 100 --atmosphere.fried_parameter 15

# generate new default yaml config based on cfg.py dataclasses, exported to ./configs/default.yaml
python cfg.py
```





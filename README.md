# Communicating Unexpectedness for Out-of-Distribution Multi-Agent Reinforcement Learning
This repository contains  the source code for our IEEE Access paper "Communicating Unexpectedness for Out-of-Distribution Multi-Agent Reinforcement Learning."

## Installation

Install dependencies.
```
conda create -n ues python=3.8.10
conda activate ues
cd epymarl
pip install pip==21.0
pip install -r requirements.txt
```

Install PyTorch from the official documentation ([link](https://pytorch.org/get-started/locally/)). Note that this repository does not rely too much on GPU, so installing for CPU should suffice.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Custom RWARE environments are located in `src/envs/robotic_warehouse`. They are registered as Gym environments and must be installed to be used.
```
cd src/envs/robotic_warehouse
pip install -e .
cd ../../..
```


## Main Experiments
To run pre-training, run the following command:
```
python src/main.py --config=<algo_config> --env-config=<env_config> with save_model=True
```
where `<algo_config>` can be one of the following:
- `ippo_pretr`: IPPO
- `ippo_r_pretr`: IPPO+M(R)
- `ippo_ues_pretr`: IPPO+M(UES)
- `ippo_uesr_pretr`: IPPO+M(UES+R)

and `<env_config>` can be one of the following:
- `doubleref_ood_train`: Pre-training environment for two-way referential game 
- `rware_orig`: Pre-training environment for RWARE

After the run is finished, `results/models` directory will contain the trained weight files. Transfer learning can be conducted via the following command:
```
src/main.py --config=<algo_config> --env-config=<env_config> with load_step=10000000 checkpoint_path=<ckpt_path>
```
where `<algo_config>` can be one of the following:
- `ippo_down`: IPPO
- `ippo_r_down`: IPPO+M(R)
- `ippo_ues_down`: IPPO+M(UES)
- `ippo_uesr_down`: IPPO+M(UES+R)

and `<env_config>` can be one of the following:
- `doubleref_ood_test_02`, `doubleref_ood_test_04.yaml`, ..., `doubleref_ood_test_10.yaml`: Downstream environments for two-way referential game, where the last two digits indicates the probability 0.2, 0.4, ..., 1.0 that the rightmost button is prompted.
- `rware_goalshift`, `rware_shelfshift`: Goal-Shift and Shelf-Shift environments for RWARE.

`<ckpt_path>` must correspond to the path of the directory that contains the pre-trained weight files.

## Plotting Learning Curves

The `plot` directory contains scripts for plotting learning curves and aggregating performances for multiple runs. To plot results for pre-training runs, use the below command:
```
python plot_pretr.py config_pretr_<env>.json
```
where `<env>` corresponds to either `rware` or `twowayref`. Prior to running this script, `run_prefix` in `plot_pretr.py` must be set to wandb project directory where the desired runs are stored. Also, the configuration file `config_pretr_<env>.json` must be edited such that the `runXid`'s are replaced with the actual IDs of the wandb runs.

To plot results for downstream runs, use the below command:
```
python plot_transf.py config_down_<env>.json
```
with changes made to the `run_prefix` and run IDs similarly to pre-training case.
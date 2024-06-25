# $\text{Alpha}^2$: Discovering Logical Formulaic Alphas using Deep Reinforcement Learning

[Link to paper](http://arxiv.org/abs/2406.16505) 

This repository contains pseudocode and algorithms for the paper "$\text{Alpha}^2$: Discovering Logical Formulaic Alphas using Deep Reinforcement Learning". It does not contain a runnnable version of $\text{Alpha}^2$, but provides the design principals and code structures. 

# Code Structure

- `utils`: utility functions for logging and loading configs
- `computation_data.py`: Generates a data file for the experiment ro run
- `run.py`: main file for running the experiment
- `run.sh`: script to start an experiment: first generate computation data, then start the runner
- `configs` configuration files
- `trainer.py`: definition of MCTS and network trainer actors for ray
- `expression` contains definition of the environment, including:
    - `evaluate.py` defines teh evaluation function
    - `legal_actions.py` calculates the legal actions when expanding an MCTS node
    - `meta_data.py` meta data of stock/futures market
    - `operands.py` definition of operands
    - `operators.py` definition of operators
    - `tokens.py` tokens wrap the implementation of operators, and implements a "validity_check" function for legal action check
    - `port.py` avoid ray recursive import
    - `structure.py` defines the structure of tokens, tree nodes, dimensions and values
    - `tree.py` defines the structure and computation of expression trees
- `mcts` contains MCTS and network related code, which is an modificated version of [alphadev](https://github.com/google-deepmind/alphadev)

# Cite this work
```bibtex
@article{xu2024textalpha2,
    title={$\text{Alpha}^2$: Discovering Logical Formulaic Alphas using Deep Reinforcement Learning},
    author={Feng Xu and Yan Yin and Xinyu Zhang and Tianyuan Liu and Shengyi Jiang and Zongzhang Zhang},
    journal={arXiv preprint arXiv:2406.16505},
    year={2024}
}
```
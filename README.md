# MultiArchy

MultiArchy is a framework for training deep multi-agent hierarchical policies. Have fun! -Brandon

# Features

We currently implement the following algorithms:

* Monolithic: SAC, TD3, PPO
* Hierarchical: HIRO, HAC, HiPPO

# Installation

Install MultiArchy by cloning the repo and using pip:

```
git clone http://github.com/brandontrabucco/multiarchy
pip install -e multiarchy
```

# Dependencies

We require a few packages for training:

* Ray 0.7.5 is tested and working
* Mujoco 2.0 should be installed on the machine running training
* TensorFlow 2.0 should be installed on the machine running training

# Research

We are actively implementing more reinforcement learning algorithms, and using MultiArchy for internal research. Refer to `multiarchy/multiarchy/baselines` for reference on how to build your own hierarchy.

<p align="center">
  <img src="https://github.com/stephenbaek/imagiqfl/raw/master/docs/images/imagiq-logo.png" width="50%" alt="imagiq-logo">
</p>

--------------------------------------------------------------------------------
![Python application](https://github.com/stephenbaek/imagiqfl/workflows/Python%20application/badge.svg)
[![codecov](https://codecov.io/gh/stephenbaek/imagiqfl/branch/master/graph/badge.svg?token=C0AT669BCM)](https://codecov.io/gh/stephenbaek/imagiqfl)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ImagiQ is an open source project to develop a usage-friendly, collaborative deep learning platform for medical imaging. It builds upon [PyTorch](https://pytorch.org/) and [MONAI](https://github.com/Project-MONAI/MONAI) ecosystem. The objectives are:

- To develop a usage-driven platform for a collaborative deep learning model development;
- To serve researchers with the standardized way to train and test deep learning models across institutions without direct patient data sharing;
- To create state-of-the-art federated learning workflows for medical imaging;

ImagiQ was funded by the Convergence Accelerator (C-Accel) Program at the National Science Foundation.

<p align="center">
  <img src="https://github.com/stephenbaek/imagiqfl/raw/master/docs/images/NSF_CA_H_LOGO_LOCKUP_BLACK.png" alt='imagiq-logo'>
</p>

# Getting Started
## Installation
### Using Anaconda/Miniconda
Probably the simplest way to get started with ImagiQ-FL is by using [Conda](https://en.wikipedia.org/wiki/Conda_(package_manager)), which can be downloaded from https://www.anaconda.com/products/individual. Installing Conda should be fairly straightforward, with lots of tutorials and blog posts you can find on the internet (such as [this](https://www.youtube.com/watch?v=YJC6ldI3hWk&ab_channel=CoreySchafer)).

Once Conda has been properly installed, you should be able to run the following command in terminal/command prompt to create a virtual environment:
```
conda create -n imagiq-fl python=3.8 ipykernel nb_conda_kernels
```
Note that `imagiq-fl` is the name of the environment to be created and `python=3.8` specifies a Python version for the environment. ImagiQ-FL has only been tested on Python version 3.8. `ipykernel` and `nb_conda_kernels` modules are installed together so that [demo](TODO:link_here) notebooks can be executed within the environment.

After the new environment is created, activate it by typing:
```
conda activate imagiq-fl
```

Next is to install [Pytorch](https://pytorch.org/). Unfortunately, steps for installing Pytorch may vary across operating systems and other system configurations. Your best bet perhaps is to try the installation guideline in the [Pytorch official website](https://pytorch.org/). Same as for Conda, there are many resources you can find on the internet explaining how to install Pytorch (such as [this](https://www.youtube.com/watch?v=vBfM5l9VK5c&ab_channel=JeffHeaton)).

Also, please install MONAI using the following code, to make sure you are using the latest version of MONAI. The version that's available from `pip install` (monai 0.3.0) has some bugs.
```
pip install git+https://github.com/Project-MONAI/MONAI#egg=MONAI
```

Finally, once Pytorch is installed, run below to install all the dependencies:
```
pip install -r requirements.txt
```

### Using VirtualEnv
TODO: Add a description here.

## Running Demos
TODO: Add a description here.

# Contributing
We welcome all types of contributions from the community, including bug fixes, feature requests, extensions, new features/functions, etc. Please feel free to open an issue and discuss your thoughts with us. To learn more about making a contribution to ImagiQ, please refer to [Contributions Page](CONTRIBUTING.md).

# Community
Join the conversation on our [Slack Channel](https://imagiq.slack.com/). 


# The Team
ImagiQ is currently maintained by [Stephen Baek](http://www.stephenbaek.com), Joseph Choi, Yankun huang, Yomi Kang, Steve Fiolic, and Jerome Charton. A non-exhaustive list of contributors includes Nick Street, Xiaodong Wu, Daniel Rubin, Paul Chang, Jayashree Kalpathy-Cramer, Sanjay Aneja, Qihang Lin, Tong Wang, Patrick Fan, Omar Chowdhury, Michael Abramoff, John Buatti, Sandeep Laroia, and Changhyun Lee.

# Citation
ImagiQ is free for an academic use. Please do not forget to cite the following paper.

TODO: Write a paper!

# License
TODO: Define a license. 

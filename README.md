## Cloud feedback from Isca PPE
Analyze the cloud feedbacks from Isca perturbed parameter ensemble (PPE) simulations

#### Dataset
* The Isca model outputs are available on Zenodo:[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5150241.svg)](https://doi.org/10.5281/zenodo.5150241). Download and unzip them into `inputs` directory.
* Download the cloud radiative kernel and observation dataset
```bash
$ ./download_dataset.sh
```
#### How to reproduce the figures

```bash
# Download the input dataset and some need to download manually from Zenodo
$ ./download_dataset.sh
$ ./runall.sh
```


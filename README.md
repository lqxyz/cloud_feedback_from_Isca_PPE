## Cloud feedback from Isca PPE
Analyze the cloud feedbacks from Isca perturbed parameter ensemble (PPE) simulations

#### Dataset
* The Isca model outputs are available on Zenodo:[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5150241.svg)](https://doi.org/10.5281/zenodo.5150241). Download and unzip them into `inputs` directory.
* Download the cloud radiative kernel, CMIP feedback and ECS, and the observational datasets. You may need to delete the symbol links in `inputs` folder before running the following command (Or just comment out the `find $outdir -type l -delete` at the beginning of the script).
```bash
$ ./download_dataset.sh
```

#### How to reproduce the figures
```bash
$ ./runall.sh
```

# 3D optical coherence refraction tomography
We have extended optical coherence refraction tomography (OCRT) to 3D by incorporating a parabolic mirror, allowing acquisition of 3D OCT volumes across two rotation axes without moving the sample. This repository includes code that registers and combines these volumes to form a resolution-enhanced, speckle-reduced, refraction-corrected 3D OCRT reconstruction along with a coregistered refractive index map of the sample.

See also our original 2D OCRT implementation: https://github.com/kevinczhou/optical-coherence-refraction-tomography/

## Datasets
The datasets for the four biological samples analyzed in the paper (fruit fly, zebrafish, mouse trachea, mouse esophagus) can be downloaded from [here](https://doi.org/10.7924/r46h4pk10). Be warned that they are rather large -- 123 GB per sample. This corresponds to 96 multi-angle OCT volumes with 400 by 400 A-scans, each with 2000 pixels (96\*400\*400\*2000\*32 bits = 122.88 GB).

I've also included checkpoint files for each sample, which contain pre-calibrated boundary conditions that are used by tensorflow to initialize the optimization variables.

## Dependencies
The code depends on the following libraries:
- tensorflow (>=2.2, gpu version preferable)
- numpy
- scipy
- matplotlib
- h5py
- jupyter

Alternatively, you can use `environment.yml` to recreate the conda environment I used, e.g.:

    conda env create --name tf2 --file=environment.yml

## Usage
Download the biological sample(s) into `/data` and run the jupyter notebook. I tested this code on an 11-GB GPU, but if your GPU is smaller (or larger), you can try adjusting `batch_size_stratified` in the second notebook cell. Unlike our [2D implementation](https://github.com/kevinczhou/optical-coherence-refraction-tomography/), this version doesn't require as much CPU RAM (but does require significantly more storage space to accommodate the 123-GB/sample datasets).

## Citation
TBD

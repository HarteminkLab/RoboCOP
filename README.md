# Papers

Mitra S., Zhong J., Tran T.Q., MacAlpine D.M., Hartemink A.J. (2021)
RoboCOP: jointly computing chromatin occupancy profiles for numerous
factors from chromatin accessibility data. Nucleic Acids Research, vol 49. https://doi.org/10.1093/nar/gkab553  

Mitra S., Zhong J., MacAlpine D.M., Hartemink A.J. (2020) RoboCOP: Multivariate State Space Model Integrating Epigenomic Accessibility Data to Elucidate Genome-Wide Chromatin Occupancy. In: Schwartz R. (eds) Research in Computational Molecular Biology. RECOMB 2020. Lecture Notes in Computer Science, vol 12074. Springer, Cham.  https://doi.org/10.1007/978-3-030-45257-5_9

# RoboCOP (Robotic Chromatin Occupancy Profiler)
---------------------------------------------------------------------------
Multivariate state space model that integrates nucleotide sequence and
chromatin accessibility data (currently uses only with MNase-seq or ATAC-seq) to
compute genome-wide probabilistic occupancy landscape of nucleosomes and
transciption factors (collectively known as DNA-binding factors or DBFs).

The list of required python and R modules can be found in [robocop-spec-file.txt](https://github.com/HarteminkLab/RoboCOP/blob/master/robocop-spec-file.txt).

### Installation:

A conda environment can be created using [robocop-spec-file.txt](https://github.com/HarteminkLab/RoboCOP/blob/master/robocop-spec-file.txt) as:

```
 conda create --name robocop-env --file robocop-spec-file.txt
 conda activate robocop-env
```

Download RoboCOP from Github.

```
 git clone https://github.com/HarteminkLab/RoboCOP.git
```

Change to RoboCOP directory and generate the shared object librobocop.so.

```
 cd RoboCOP
 cd robocop
 bash gccCompile # generates shared library librobocop.so
 ls "`pwd`/librobocop.so" # display absolute path of librobocop.so; copy path
```

Add path of shared library to your configuration file. Example
configuration file provided in analysis directory ([config_example.ini](https://github.com/HarteminkLab/RoboCOP/blob/master/analysis/config_example.ini)). So from the robocop
directory:

```
 cd ../../analysis/
```

Open file config_example.ini in an editor and paste path to cshared. For example if
path is /home/myhome/RoboCOP/pkg/robocop/librobocop.so then in config.ini set:

```
 cshared=/home/myhome/RoboCOP/pkg/robocop/librobocop.so
```

### [Tutorial](https://github.com/HarteminkLab/RoboCOP/blob/master/analysis/example_robocop.ipynb)

Introductory vignette describing how to run RoboCOP can be found
[here](https://github.com/HarteminkLab/RoboCOP/blob/master/analysis/example_robocop.ipynb)
in a
jupyter notebook format in the [analysis](https://github.com/HarteminkLab/RoboCOP/tree/master/analysis) directory.

Vignette describing how to run DynaCOP can be found [here](https://github.com/HarteminkLab/RoboCOP/blob/master/analysis/example_dynacop.ipynb) in a jupyter
notebook format in the [analysis](https://github.com/HarteminkLab/RoboCOP/tree/master/analysis) directory.

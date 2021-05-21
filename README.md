RECOMB paper: Mitra S., Zhong J., MacAlpine D.M., Hartemink A.J. (2020) RoboCOP: Multivariate State Space Model Integrating Epigenomic Accessibility Data to Elucidate Genome-Wide Chromatin Occupancy. In: Schwartz R. (eds) Research in Computational Molecular Biology. RECOMB 2020. Lecture Notes in Computer Science, vol 12074. Springer, Cham.  https://doi.org/10.1007/978-3-030-45257-5_9

# RoboCOP (Robotic Chromatin Occupancy Profiler)
---------------------------------------------------------------------------
Multivariate state space model that integrates nucleotide sequence and
chromatin accessibility data (currently used only with MNase-seq and/or ATAC-seq) to
compute genome-wide probabilistic occupancy landscape of nucleosomes and
transciption factors (collectively known as DNA-binding factors or DBFs).

Python requirements:
- python 3.6+
- numpy
- pandas
- scipy
- ctypes
- roman
- pysam
- rpy2
- biopython
- matplotlib (only for plotting -- optional)

R requirements:
- MASS
- fitdistrplus

### Installation:

Python packages can be installed using pip3. R packages can be installed
through install.packages() in R console.

Change to RoboCOP directory and install

```
 cd RoboCOP
 python setup.py install
 cd robocop
 chmod +x gccCompile
 ./gccCompile # generates shared library librobocop.so
 ls "`pwd`/librobocop.so" # display absolute path of librobocop.so; copy path
```

Add path of shared library to your configuration file. Example
configuration file provided in analysis directory. So from the robocop
directory:

```
 cd ../../analysis/
```

Open file config.ini in an editor and paste path to cshared. For example if
path is /home/myhome/RoboCOP/pkg/robocop/librobocop.so then in config.ini set:

```
 cshared=/home/myhome/RoboCOP/pkg/robocop/librobocop.so
```

### Download MNase-seq BAM file

Update path of mnaseFile in config.ini to the path of the MNase-seq BAM file you want to use.

An example MNase-seq BAM file that we used in the RECOMB submission can be downloaded from <https://doi.org/10.7924/r4hx1b43s>.

### Running on a single MNase-seq or ATAC-seq file:

Have the path of all configuration files in config.ini. To run RoboCOP on a
set of genome regions with Baum-Welch update of transition probabilities:

```
python robocop_em.py <coordinates file -- example analysis/coordinates.bed> <config file -- example analysis/config.ini> <output directory -- example OutDir>
```

It is better to run robocop_em.py on a small set of coordinates and then
use the learned parameters to perform posterior decoding on larger
genomic regions.

```
 python robocop_no_em.py <coordinates file> <config file> <output directory with learned parameters -- example OutDir> <new output directory -- example NewOutDir>
```

# RoboCOP (Robotic Chromatin Occupancy)
---------------------------------------------------------------------------
Multivariate state space model that integrates nucleotide sequence, and
chromatin accessibility data (currently used only with MNase-seq) to
compute genome-wide probabilistic occupancy landscape of nucleosomes and transciption
factors, collectively known as DNA binding factors or DBFs.

Python Requirements:
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

R requirement:
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

### Download MNase-seq BAM file -- TO DO

Update path of files in $$config.ini$$.

### Running:

Have the path of all configuration files in config.ini. To run RoboCOP on a
set of genome regions with Baum-Welch update of transition probabilities:

```
python robocop_em.py <coordinates file -- example analysis/coordinates.bed> <config file -- example analysis/config.ini> <output directory -- OutDir>
```

It is better to run robocop_em.py on a small set of coordinates and then
use the learned parameters to perform posterior decoding on larger
genomic regions.

```
 python robocop_no_em.py <coordinates file> <config file> <output directory with learned parameters -- example OutDir> <new output directory -- example NewOutDir>
```

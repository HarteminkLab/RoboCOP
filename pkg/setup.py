#!/usr/bin/env python

from setuptools import setup, Extension

setup(
    name='robocop',
    version='2.0',
    description='robocop, a multivariate hidden Markov model designed to infer regulatory protein binding probability along the genome using nucleotide sequence and MNase-seq.',
    author='Sneha Mitra and Jianling Zhong',
    author_email='sneha@cs.duke.edu',
    packages=[ 'robocop', 'robocop.nucleosome', 
                'robocop.utils'],
    package_data = {'': ['*.txt', '*.fasta', '*.gff', '*.csv',
                          'sacCer2/*.fasta', 'sacCer2/ascii/*.txt', '*.gff', '*.so']},
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Academic',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Statistics',
    ],

)



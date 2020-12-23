import glob
import numpy as np

if __name__ == '__main__':
    allfiles = glob.glob("/usr/xtmp/sneha/compete_16_Chr1_16/tmpDir/posterior_and_emission.idx*.npz")
    print("Not present:")
    alli = []
    for i in range(3004):
        fname = "/usr/xtmp/sneha/compete_16_Chr1_16/tmpDir/posterior_and_emission.idx" + str(i) + ".npz"
        if fname not in allfiles:
            print(fname)
            alli.append(str(i))

    print(",".join(alli))

    for i in range(1732, 3004):
        fname = "/usr/xtmp/sneha/compete_16_Chr1_16/tmpDir/posterior_and_emission.idx" + str(i) + ".npz"
        ptable = np.load(fname, allow_pickle = True)['posterior']
        if ptable is None:
            print(i)

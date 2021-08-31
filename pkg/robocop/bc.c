#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> // memset is there
#include "bc.h"

int I3(int z, int row, int nrow, int col, int ncol)
{
  // returns linearized index of the matrix from index [z, row, col]
  // each z has nrow * ncol elements
  return z*nrow*ncol + row*ncol + col;
}

int I(int row, int col, int ncol) {
    // returns the linerized index of the matrix form index [i,j]
    // each row of the matrix has ncol elements
    return row * ncol + col;
}

int Ir(int row, int col, int nrow, int ncol) {
    // returns the linerized index of the matrix form index [i,j]
    // each row of the matrix has ncol elements
    return (nrow - 1 - row) * ncol + col;
}

void *ALLOC(size_t size) {
  void *foo;
  if(!(foo = malloc(size))) {
    fprintf(stderr, "malloc failed.\n");
    exit(1);
  }

  return foo;
}


void construct_transition_matrix(int n_states, int nuc_present, int nuc_start, long* motif_starts, long* motif_lens, int n_motifs, int silent_states_begin,
  // output
  double* transition_matrix, char* nuc_dinucleotide_model_file
  ) 

{
    int i,j;    
    // we need to set up the skeleton of the transition matrix
    // 1. set transition within motifs to be 1
    // 2. set the dinucleotide transition of nucleosome model, if present
    // 3. set transition back to central silent state to be 1
    
    // background state
    transition_matrix[I(0, silent_states_begin, n_states)] = 1.0;
    // tf states
    for(i = 0; i < n_motifs; i++) {
        // forward motif
        for(j = motif_starts[i]; j < motif_starts[i] + motif_lens[i] - 1; j++) {
            transition_matrix[I(j, j+1, n_states)] = 1.0;
        }
            
        transition_matrix[I(j, silent_states_begin, n_states)] = 1.0;
        
        // advance to the reverse motif start
        j += 1;
        // reverse motif
        for(; j < motif_starts[i] + 2 * motif_lens[i] - 1; j++) {
            transition_matrix[I(j, j+1, n_states)] = 1.0;
        }


        transition_matrix[I(j, silent_states_begin, n_states)] = 1.0;
        
        // The tf silent states to the start of forward and reverse motifs
        transition_matrix[
            I(silent_states_begin + i+1, motif_starts[i], n_states)
                ] = 0.5;

        transition_matrix[
            I(
            silent_states_begin + i+1, motif_starts[i] + motif_lens[i], n_states
            )
                ] = 0.5;
    }
    
    if(nuc_present) {
        int from, to;
        float prob;
        char line[512];
        FILE* nuc_model_file;
        
        nuc_model_file = fopen(nuc_dinucleotide_model_file, "r");
        if(nuc_model_file == NULL) {
            fprintf(stderr, "Can't open nucleosome dinucleotide model file: %s\n", nuc_dinucleotide_model_file);
            exit(1);
        }
        while(fgets(line, 512, nuc_model_file)) {
            if(line[0] != '#') {
                sscanf(line, "%d %d %f", &from, &to, &prob);
                transition_matrix[
                    I(nuc_start + from - 1, nuc_start + to - 1, n_states)
                        ] = prob;
            }
        }
            
        
        // last nucleosome state to central silent state
        transition_matrix[I(nuc_start+to-1, silent_states_begin,n_states)] = 1.0;
        fclose(nuc_model_file);
    }
    
}

int build_emission_mat_from_pwm(
    // input
    long *sequence, double *pwm, 
    // info about input
    int n_obs, int n_states, int silent_states_begin, int alphabet_length,
    // output
    double *emission_mat
    )
{
    int i,j;

    for(i = 0; i < n_obs; i++) {
        for(j = 0; j < silent_states_begin; j++) {
            emission_mat[I(i, j, n_states)] = pwm[j * alphabet_length + sequence[i]];
        }
        for(j = silent_states_begin; j < n_states; j++) {
            emission_mat[I(i, j, n_states)] = 1.0;
        }
    }

    return 0;
}


void set_initial_probs( long *motif_starts, long *motif_lens,
                        int n_motifs, int silent_states_begin,
                        int n_states, 
                        int nuc_present, int nuc_start, int nuc_len,
                        double* transition_matrix,
                        // output
                        double* initial_probs
                        ) {

  int i, j;
  int n_nuc_state_per_position;
  double sum = 0;

  int only_allow_start_from_motif_1st_position = 1;
  
  // start with background component, which is only 1 long...
  sum = transition_matrix[I(silent_states_begin, 0, n_states)];
  initial_probs[0] = sum;
  
  for (i = 0; i < n_motifs; i++) {
    double p = transition_matrix[I(silent_states_begin, silent_states_begin + i + 1, n_states)];

    if(only_allow_start_from_motif_1st_position) {
      j = motif_starts[i];
      initial_probs[j] = p / 2.0; // half for the forward motif and half for the reverse motif
      sum += initial_probs[j];
      
      for(j = motif_starts[i] + 1; j < motif_starts[i] + motif_lens[i]; j++)
          initial_probs[j] = 0;

      j = motif_starts[i] + motif_lens[i];
      initial_probs[j] = p / 2.0; // half for the forward motif and half for the reverse motif
      sum += initial_probs[j];

      for(j = motif_starts[i] + motif_lens[i] + 1; j < motif_starts[i] + 2 * motif_lens[i]; j++)
        initial_probs[j] = 0;
    }
    else {
      for (j = motif_starts[i]; j < motif_starts[i] + 2 * motif_lens[i]; j++) {
        initial_probs[j] = p;
        sum += p;
      }  
    }
  }
  
  if (nuc_present) {
    double p = transition_matrix[I(silent_states_begin, nuc_start, n_states)];

    if(only_allow_start_from_motif_1st_position) {
        initial_probs[nuc_start] = p;
        sum += p;
        for(i = nuc_start + 1; i < nuc_start + nuc_len; i++)
          initial_probs[i] = 0;
    } else {
      
      int n_padding_states = 18; // There are 18 padding states at the begining of nucleosome in current dinucleotide nucleosome model (14 normal background states plus 4 branching background states).

      // left (normal) padding states
      for (i = nuc_start; i < nuc_start + n_padding_states - 4; i++) {
        initial_probs[i] = p;
        sum += p;
      }

      // branched padding state
      for (i = nuc_start + n_padding_states - 4; i < nuc_start + n_padding_states; i++) {
        initial_probs[i] = p / 4.0;
        sum += p / 4.0;
      }

      // tons of nucleosome states, 16 or 4 per sequence position
      // if 4 per nucleotide position, then the total nuc_len should be 541
      // if 16 per sequence, nuc_len should be more than 1000 (forgot exact number...)
      if(nuc_len < 1000) {
        n_nuc_state_per_position = 4;
      } else {
        n_nuc_state_per_position = 16;
      }
      
      for (i = nuc_start + n_padding_states; i < nuc_start + nuc_len - n_padding_states + 3; i++) {
        initial_probs[i] = p / n_nuc_state_per_position;
        sum += p / n_nuc_state_per_position;
      }

      // right branching states, all of which are normal
      for (i = nuc_start + nuc_len - n_padding_states + 3; i < nuc_start + nuc_len; i++) {
        initial_probs[i] = p;
        sum += p;
      }
    assert(i == silent_states_begin);      
    }

  }
  
}


void find_parents_and_children(long* parents, long* children, long* n_parents, long* n_children, int n_states, int silent_states_begin, double* transition_matrix) {

  int i, j;

  for (i = 0; i < n_states; i++) {

    n_parents[i] = n_children[i] = 0;

    for (j = 0; j < n_states; j++) {
      if (transition_matrix[I(j, i, n_states)] != 0.0) {
        parents[I(i, n_parents[i], n_states)] = j;
        n_parents[i]++;
      }
      if (transition_matrix[I(i, j, n_states)] != 0.0) {
        children[I(i, n_children[i], n_states)] = j;
        n_children[i]++;
      }
    }

  }
}




// Some helper functions for debugging 
void print2dmatrix(double* table, int nrow, int ncol) {
    int i,j;
    for(i = 0; i < nrow; i++) {
        for(j = 0; j < ncol - 1; j++) {
            fprintf(stderr, "%.2f\t", table[I(i, j, ncol)]);
        }
        fprintf(stderr, "%.2f\n", table[I(i, ncol - 1, ncol)]);
    }
}


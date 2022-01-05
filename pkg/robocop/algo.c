#include <math.h>
#include <float.h>
#include <assert.h>
#include <gsl/gsl_randist.h>
# include "bc.h"


double normalize(
    // normalizing one row in the table
    double *table, 
    // info about the table
    int row, int ncol,
    // when calculating normalizing constant, sum to idx
    int idx
    ) 
{
    double sum = 0;
    int i;
    for(i = 0; i < idx; i++) {
        sum += table[I(row, i, ncol)] + DBL_MIN;
    }

    for(i = 0; i < ncol; i++) {
      table[I(row, i, ncol)] = (table[I(row, i, ncol)] + DBL_MIN) / sum;
    }

    return sum;
}


int fward(
    // input
    double *initial_probs,
    double *transition_mat, double *emission_mat, 
    double *end_probs,
    // information about the matrices:
    int n_states, int silent_states_begin, int n_obs, int n_vars,
    long *parents_mat, long *n_parents,

    // information about the motif positions
    long *motif_starts, long *motif_lens, int n_motifs,
    int nuc_present, int nuc_start, int nuc_len,

    // output
    double *ftable, double *scaling_factors
    ) 
{

  int i, j, k, l;
    int parent;
    
    // initialization. 
    
    for(j = 0; j < silent_states_begin; j++) {
      //        ftable[I(0, j, n_states)] = initial_probs[j] * emission_mat[I(0, j, n_states)];
      ftable[I(0, j, n_states)] = initial_probs[j];
      for(k = 0; k < n_vars; k++)
	ftable[I(0, j, n_states)] *= emission_mat[I3(k, 0, n_obs, j, n_states)];
      
    }

    for(j = silent_states_begin; j < n_states; j++) {
        for(k = 0; k < n_parents[j]; k++) {
            parent = parents_mat[I(j, k, n_states)];
            //parent = parents_mat[j][k];
            ftable[I(0, j, n_states)] += ftable[I(0, parent, n_states)] * transition_mat[I(parent, j, n_states)];
        }
    }

    
    // nomalization
    scaling_factors[0] = normalize(ftable, 0, n_states, silent_states_begin);

    // fill in the rest of the table
    // the last position should be filled differently because certain state 
    // should not appear at the end
    for(i = 1; i < n_obs-1; i++) {
        for(j = 0; j < silent_states_begin; j++) {
            for(k = 0; k < n_parents[j]; k++) {
                parent = parents_mat[I(j, k, n_states)];
                //parent = parents_mat[j][k];
                ftable[I(i, j, n_states)] += ftable[I(i - 1, parent, n_states)] * transition_mat[I(parent, j, n_states)];
            }
	    for(l = 0; l < n_vars; l++)
	      ftable[I(i, j, n_states)] *= emission_mat[I3(l, i, n_obs, j, n_states)];

        }

        // silent states
        for(j = silent_states_begin; j < n_states; j++) {
            for(k = 0; k < n_parents[j]; k++) {
                parent = parents_mat[I(j, k, n_states)];
                // parent = parents_mat[j][k];
                ftable[I(i, j, n_states)] += ftable[I(i, parent, n_states)] * transition_mat[I(parent, j, n_states)];
            }
        }

        scaling_factors[i] = normalize(ftable, i, n_states, silent_states_begin);
    }

    // fill the last position
    i = n_obs-1;
    for(j = 0; j < silent_states_begin; j++) {
        for(k = 0; k < n_parents[j]; k++) {
            parent = parents_mat[I(j, k, n_states)];
            //parent = parents_mat[j][k];
            ftable[I(i, j, n_states)] += ftable[I(i - 1, parent, n_states)] * transition_mat[I(parent, j, n_states)];
        }
	ftable[I(i, j, n_states)] *= end_probs[j];
	for(l = 0; l < n_vars; l++)
	  ftable[I(i, j, n_states)] *= emission_mat[I3(l, i, n_obs, j, n_states)];
    }

    // silent states
    for(j = silent_states_begin; j < n_states; j++) {
        for(k = 0; k < n_parents[j]; k++) {
            parent = parents_mat[I(j, k, n_states)];
            // parent = parents_mat[j][k];
            ftable[I(i, j, n_states)] += ftable[I(i, parent, n_states)] * transition_mat[I(parent, j, n_states)];
        }
    }

    scaling_factors[i] = normalize(ftable, i, n_states, silent_states_begin);

    return 0;
}


int bward(
    double *transition_mat, double *emission_mat, 
    double *end_probs,
    // information about the matrices:
    int n_states, int silent_states_begin, int n_obs, int n_vars,
    long *children_mat, long *n_children,
    long *motif_starts, long *motif_lens, int n_motifs,
    int nuc_present, int nuc_start, int nuc_len,
    // output, backward table is stored in reverse order
    // for faster memory access
    double *btable, double *scaling_factors
    ) 
{
    
  int i, j, k, l;
    int child;
    double emissionValues;
    
    // initialization 

    for(j = 0; j < silent_states_begin; j++) {
        btable[I(0, j, n_states)] = end_probs[j];
    }


    for(j = n_states - 1; j > silent_states_begin - 1; j--) {
        for(k = 0; k < n_children[j]; k++) {
            //child = children_mat[j][k];
            child = children_mat[I(j,k,n_states)];
	    emissionValues = 1;
	    for(l = 0; l < n_vars; l++) emissionValues *= emission_mat[I3(l, n_obs - 1, n_obs, child, n_states)];
	    btable[I(0, j, n_states)] += btable[I(0, child, n_states)] * transition_mat[I(j, child, n_states)] * emissionValues;
	    

        }
    }

    scaling_factors[n_obs - 1] = normalize(btable, 0, n_states, silent_states_begin);

    // the rest of the table
    for(i = 1; i < n_obs; i++) {
        for(j = 0; j < silent_states_begin; j++) {
            for(k = 0; k < n_children[j]; k++) {
                //child = children_mat[j][k];
                child = children_mat[I(j,k,n_states)];
		emissionValues = 1;
		for(l = 0; l < n_vars; l++) emissionValues *= emission_mat[I3(l, n_obs - i, n_obs, child, n_states)];
                btable[I(i, j, n_states)] += 
                    btable[I(i - 1, child, n_states)] * 
                    transition_mat[I(j, child, n_states)] * 
		  emissionValues;
		

            }
        }

        for(j = n_states - 1; j > silent_states_begin - 1; j--) {
            for(k = 0; k < n_children[j]; k++) {
                //child = children_mat[j][k];
	      child = children_mat[I(j,k,n_states)];
	      emissionValues = 1;
	      for(l = 0; l < n_vars; l++) emissionValues *= emission_mat[I3(l, n_obs - i - 1, n_obs, child, n_states)];

                btable[I(i, j, n_states)] += 
                    btable[I(i, child, n_states)] * 
                    transition_mat[I(j, child , n_states)] *
                    emissionValues;

            }
        }


        scaling_factors[n_obs - 1 - i] = normalize(btable, i, n_states, silent_states_begin);
    }

    return 0;
}

double calc_sr(double *sf, double *sb, int len, long double *sr) {
  int i;
  double log_sf_sum  = 0;

  sr[len - 1] = 1;
//  sr[len - 1] = log(1);
  for (i = len - 1; i > 0; i--) {
    sr[i - 1] = sr[i] * (sb[i] / sf[i]) ;
    
    // if(isinf(sr[i - 1])) sr[i - 1] = DBL_MAX;

    log_sf_sum += log(sf[i]);
//    sr[i - 1] = sr[i] + log(sb[i]) - log(sf[i]);
  }

  log_sf_sum += log(sf[0]);

  return log_sf_sum;
}

void posterior_decoding(
    double *f_table, double *b_table, 
    double *sb, long double *sr, 
    int n_states, int n_obs,
    // output
    double* posterior_table
    ) {

   int i,j;

   for(i = 0; i < n_obs; i++) {
    for(j = 0; j < n_states; j++) {
        posterior_table[I(i,j,n_states)] = 
	  (sb[i] * b_table[Ir(i,j,n_obs,n_states)]) * (sr[i] * f_table[I(i,j,n_states)]);
    }
   }
}

void update_data_emission_matrix_with_discrete_pmf(double* pmf, long* data, long max_count, long nrow, long ncol, long col_start, long col_end, double* emission_matrix) {

    // emission_matrix[I(i,j)] = P(data_i | state_j)
    // emission_matrix is stroed as row continuous
    int i,j;

    for(i = 0; i < nrow; i++) {
        for(j = col_start; j < col_end; j++) {
            emission_matrix[I(i,j,ncol)] *= pmf[data[i]];
        }
    }
}


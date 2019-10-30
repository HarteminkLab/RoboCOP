#ifndef _bc_h
#define _bc_h

int I3(int, int, int, int, int);
int I(int row, int col, int ncol);
int Ir(int, int, int, int);
void *ALLOC(size_t);
void construct_transition_matrix(int, int, int, long*, long*, int, int, double*, char*);
int build_emission_mat_from_pwm(long *, double*, int, int, int, int, double*);
void set_initial_probs( long*, long*, int, int, int, int, int, int, double*, double*);
void find_parents_and_children(long*, long*, long*, long*, int, int, double*);
void print2dmatrix(double*, int, int);

#endif

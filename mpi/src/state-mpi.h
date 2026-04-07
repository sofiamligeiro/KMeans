#ifndef STATE_MPI_H
#define STATE_MPI_H

#include "docs-mpi.h"

#define ALLREDUCE_BUF_SIZE(C, S) ((C) * (S) + (C))

/*
 * Unified buffer and state for cluster statistics.
 *
 * This structure holds both the centroids and the document counts.
 * It is designed around a single contiguous memory block ('values')
 * so that we can synchronize all state across processes using a
 * single MPI_Allreduce call.
 */
typedef struct {
    /*
     * The master contiguous memory block for MPI operations.
     * Dimensions: double[C][S] (centroids) + double[C] (doc_counts).
     */
    double *values;

    /*
     * 2D mapping of cabinet score averages for distance calculations.
     * Transitions from local partial sums to global synchronized values post-Allreduce.
     * NOTE: Row pointers map directly into the 'values' block.
     */
    double **centroids;

    /*
     * Document count per cabinet
     * Transitions from local partial sums to global synchronized values post-Allreduce.
     * NOTE: Typed as 'double' instead of 'int' to share the contiguous 'values' block.
     * This allows one unified MPI_DOUBLE reduction for everything.
     */
    double *doc_counts;

} ClusterStats;


/*
 * State container for the algorithm's memory allocations.
 *
 * Owns all memory required by the algorithm beyond the dataset itself.
 * Allocated once before the main loop and freed once after, ensuring
 * no dynamic allocation occurs per iteration.
 */
typedef struct {
    /*
     * Local cabinet assignment for each of this process's documents.
     * Dimensions: int[local_num_documents]
     */
    int *assignment;

    /*
     * Unified buffer containing centroid and document count statistics.
     */
    ClusterStats stats;

} AlgorithmState;

AlgorithmState *init_algorithm_state(const Dataset *data, int rank, int size);

void free_algorithm_state(AlgorithmState *state);

void print_algorithm_state(const AlgorithmState *state, const Dataset *dataset, int rank);

#endif /* STATE_MPI_H */
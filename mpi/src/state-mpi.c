/**
 * Parallel and Distributed Computing 25/26
 *
 * GROUP 33:
 * Artur Krystopchuk (104145) - arturkrystopchuk@tecnico.ulisboa.pt
 * João Martins (106819) - joao.bernardo.mota.martins@tecnico.ulisboa.pt
 * Sofia Ligeiro (116046) - sofiamligeiro@tecnico.ulisboa.pt
 *
 * Description: Manages memory allocation and state initialization for
 * the algorithm. Responsible for setting up the initial document
 * assignments and all auxiliar variables that will be needed.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "state-mpi.h"
#include "parser-mpi.h"

/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */

static ClusterStats alloc_reduction_buffer(int C, int S);

/* =================================== *
 * PUBLIC FUNCTION DEFINITIONS         *
 * =================================== */

/**
 * Allocates and fully initialises an AlgorithmState for the given dataset.
 *
 * Initial cabinet assignment follows round-robin:
 * assignment[d] = (local_start + d) % C
 *
 * All other variables stats.centroids and stats.doc_counts are zeroed.
 *
 * @param data  Pointer to the local dataset partition
 * @param rank  MPI rank of the calling process
 * @param size  Total number of MPI processes
 * @return      Pointer to a fully initialised AlgorithmState, or NULL on error
 */
AlgorithmState *init_algorithm_state(const Dataset *data, int rank, int size) {
    int C = data->num_cabinets;
    int D = data->num_documents;
    int S = data->num_subjects;

    AlgorithmState *state = (AlgorithmState *)calloc(1, sizeof(AlgorithmState));
    if (!state) return NULL;

    /* --- Step 1 - round-robin initialization --- */
    int local_start, local_num_documents;
    compute_local_range(D, rank, size, &local_start, &local_num_documents);

    state->assignment = (int *)malloc(local_num_documents * sizeof(int));
    if (!state->assignment) goto cleanup;

    for (int d = 0; d < local_num_documents; d++)
        state->assignment[d] = (local_start + d) % C;

    /* --- Allocation of the buffer --- */
    state->stats = alloc_reduction_buffer(C, S);
    if (!state->stats.values) goto cleanup;

    return state;

cleanup:
    fprintf(stderr, "Rank %d: Failed to allocate AlgorithmState\n", rank);
    free_algorithm_state(state);
    return NULL;
}

/**
 * Frees all memory owned by an AlgorithmState struct.
 *
 * @param state Pointer to the state to free
 */
void free_algorithm_state(AlgorithmState *state) {
    if (!state) return;

    free(state->assignment);

    free(state->stats.centroids); /* pointer array */
    free(state->stats.values);    /* data block    */

    free(state);
}

/* =================================== *
 * INTERNAL HELPER DEFINITIONS         *
 * =================================== */

/**
 * Allocates a ReductionBuffer as a single contiguous block and wires
 * the named pointers into it. All values are zeroed on allocation.
 *
 * @param C  Number of cabinets
 * @param S  Number of subjects
 * @return   Fully wired ReductionBuffer, or a zeroed one on failure
 */
static ClusterStats alloc_reduction_buffer(int C, int S) {
    /* 1. Allocate the contiguous data block: double[C][S] (centroids) + double[C] (doc_counts) */
    double *block = (double *)calloc(ALLREDUCE_BUF_SIZE(C, S), sizeof(double));
    if (!block) return (ClusterStats){0};

    /* 2. Allocate the row-pointer array for centroids */
    double **rows = (double **)malloc(C * sizeof(double *));
    if (!rows) {
        free(block);
        return (ClusterStats){0};
    }

    /* 3. Wire each row pointer into the contiguous block */
    for (int c = 0; c < C; c++) {
        rows[c] = &block[c * S];
    }

    return (ClusterStats) {
        .values     = block,
        .centroids  = rows,
        .doc_counts = block + (C * S),
    };
}

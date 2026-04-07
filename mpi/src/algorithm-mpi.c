/**
 * Parallel and Distributed Computing 25/26
 *
 * GROUP 33:
 * Artur Krystopchuk (104145) - arturkrystopchuk@tecnico.ulisboa.pt
 * João Martins (106819) - joao.bernardo.mota.martins@tecnico.ulisboa.pt
 * Sofia Ligeiro (116046) - sofiamligeiro@tecnico.ulisboa.pt
 *
 * Description: Core logic for the distributed Document Classification algorithm.
 * Manages the iterative process of computing local centroid sums, synchronizing
 * global centroids across processes, and reassigning local documents to cabinets.
 */

#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "parser-mpi.h"
#include "state-mpi.h"
#include "algorithm-mpi.h"

/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */

static void   accumulate_partial_sums(const Dataset *data, AlgorithmState *state);
static void   normalize_centroids(AlgorithmState *state, int C, int S);
static double compute_distance(const double *doc_scores, const double *centroid, int S);
static int    reassign_local_docs(const Dataset *data, AlgorithmState *state);
//static void   debug(const Dataset *data, const AlgorithmState *state, int rank, int size);

/* =================================== *
 * PUBLIC FUNCTION DEFINITIONS         *
 * =================================== */

/**
 * Runs the distributed document classification algorithm.
 *
 * Each process operates on its local document partition. In each iteration:
 *      1. Local centroid sums and local doc counts are accumulated
 *         directly into state->stats.
 *      2. A single MPI_Allreduce(MPI_IN_PLACE) synchronizes state->stats globally.
 *      3. Centroids are normalized in-place using the synchronized doc counts.
 *      4. Local documents are reassigned to their nearest centroid.
 *      5. A second MPI_Allreduce checks whether any process moved a document.
 *
 * The loop terminates when no process reassigns any document.
 *
 * @param data  Pointer to the local dataset partition
 * @return      Pointer to the AlgorithmState holding local assignments,
 * or NULL on allocation failure
 */
AlgorithmState *classify_docs(const Dataset *data) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    AlgorithmState *state = init_algorithm_state(data, rank, size);
    if (!state) return NULL;

    int C = data->num_cabinets;
    int S = data->num_subjects;

    int global_moved;
    do {
        /* Step 1 */
        accumulate_partial_sums(data, state);
        MPI_Allreduce(MPI_IN_PLACE, state->stats.values, ALLREDUCE_BUF_SIZE(C, S),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        normalize_centroids(state, C, S);

        /* Step 2 */
        int local_moved = reassign_local_docs(data, state);
        MPI_Allreduce(&local_moved, &global_moved,
                      1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    } while (global_moved);

    return state;
}

/**
 * Prints the full global assignment array to stdout (one cabinet ID per line).
 * Should only be called by rank 0 after gathering all local assignments.
 *
 * @param assignment  Global assignment array of length D
 * @param D           Total number of documents
 */
void print_assignment(const int *assignment, int D) {
    for (int i = 0; i < D; i++)
        fprintf(stdout, "%d\n", assignment[i]);
}

/* =================================== *
 * INTERNAL HELPER DEFINITIONS         *
 * =================================== */

/**
 * Computes local partial sums of centroids and document counts.
 *
 * @param data   Pointer to the local dataset partition
 * @param state  Pointer to the current algorithm state
 */
static void accumulate_partial_sums(const Dataset *data, AlgorithmState *state) {
    int C       = data->num_cabinets;
    int S       = data->num_subjects;
    int local_D = data->local_num_documents;

    memset(state->stats.values, 0, ALLREDUCE_BUF_SIZE(C, S) * sizeof(double));

    for (int d = 0; d < local_D; d++) {
        int cab = state->assignment[d];
        state->stats.doc_counts[cab]++;

        for (int s = 0; s < S; s++)
            state->stats.centroids[cab][s] += data->documents_scores[d][s];
    }
}

/**
 * Averages the globally synchronized centroid sums.
 *
 * Divides each cabinet's centroid sum by its global document count.
 * Safely ignores empty cabinets to prevent division by zero.
 *
 * @param state  Pointer to the current algorithm state
 * @param C      Number of cabinets
 * @param S      Number of subjects per document
 */
static void normalize_centroids(AlgorithmState *state, int C, int S) {
    for (int c = 0; c < C; c++) {
        if (state->stats.doc_counts[c] == 0) continue;

        double inv_count = 1.0 / state->stats.doc_counts[c];
        for (int s = 0; s < S; s++)
            state->stats.centroids[c][s] *= inv_count;
    }
}

/**
 * Computes the squared Euclidean distance between a document and a centroid.
 *
 * @param doc_scores  Score vector for a single document
 * @param centroid    Centroid score vector for a cabinet
 * @param S           Number of subjects
 * @return            Squared Euclidean distance
 */
static double compute_distance(const double *doc_scores, const double *centroid, int S) {
    double sum = 0.0;
    for (int s = 0; s < S; s++) {
        double diff = doc_scores[s] - centroid[s];
        sum += diff * diff;
    }
    return sum;
}

/**
 * Reassigns each local document to its nearest centroid.
 *
 * For each local document, the distance to every cabinet centroid is computed.
 * If a closer cabinet is found, the document is reassigned.
 *
 * @param data   Pointer to the local dataset partition
 * @param state  Pointer to the current algorithm state
 * @return       1 if at least one document changed cabinet, 0 otherwise
 */
static int reassign_local_docs(const Dataset *data, AlgorithmState *state) {
    int C       = data->num_cabinets;
    int S       = data->num_subjects;
    int local_D = data->local_num_documents;

    int moved = FALSE;

    for (int d = 0; d < local_D; d++) {
        int    best_cabinet  = state->assignment[d];
        double best_distance = compute_distance(data->documents_scores[d],
                                                state->stats.centroids[best_cabinet], S);

        for (int c = 0; c < C; c++) {
            if (c == best_cabinet) continue;

            double distance = compute_distance(data->documents_scores[d],
                                               state->stats.centroids[c], S);
            if (distance < best_distance) {
                best_distance = distance;
                best_cabinet  = c;
            }
        }

        if (best_cabinet != state->assignment[d]) {
            state->assignment[d] = best_cabinet;
            moved = TRUE;
        }
    }

    return moved;
}


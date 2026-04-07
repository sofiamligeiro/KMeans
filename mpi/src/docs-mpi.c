/**
 * Parallel and Distributed Computing 25/26
 * 
 * GROUP 33:
 * Artur Krystopchuk (104145) - arturkrystopchuk@tecnico.ulisboa.pt
 * João Martins (106819) - joao.bernardo.mota.martins@tecnico.ulisboa.pt
 * Sofia Ligeiro (116046) - sofiamligeiro@tecnico.ulisboa.pt
 *
 * Description: Main entry point for the MPI implementation. Handles MPI
 * environment setup, coordinates dataset loading, measures execution time,
 * and gathers the final document assignments to print to stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "docs-mpi.h"
#include "parser-mpi.h"
#include "state-mpi.h"
#include "algorithm-mpi.h"

/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */

static void print_exec_time(double t_start, double t_end, int rank);
static void gather_and_print_assignment(const Dataset *data,
                                        const AlgorithmState *state,
                                        int rank, int size);

/* =================================== *
 * MAIN                                *
 * =================================== */

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    Dataset *data = load_dataset(argv[1], rank, size);
    if (!data) {
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t_start = omp_get_wtime();
    AlgorithmState *state = classify_docs(data);
    double t_end = omp_get_wtime();

    print_exec_time(t_start, t_end, rank);
    gather_and_print_assignment(data, state, rank, size);

    free_algorithm_state(state);
    free_dataset(data);

    MPI_Finalize();
    return 0;
}

/* =================================== *
 * INTERNAL HELPER DEFINITIONS         *
 * =================================== */

/**
 * Reduces each process's local execution time to the global maximum
 * (true wall-clock duration) and prints it to stderr from rank 0.
 *
 * @param t_start  Time when this process started the algorithm
 * @param t_end    Time when this process finished the algorithm
 * @param rank     MPI rank of the calling process
 */
static void print_exec_time(double t_start, double t_end, int rank) {
    double global_start, global_end;
    MPI_Reduce(&t_start, &global_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_end,   &global_end,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        fprintf(stderr, "%.1fs\n", global_end - global_start);
}

/**
 * Gathers local assignment arrays from all processes to rank 0
 * and prints the full global assignment to stdout.
 *
 * Uses MPI_Gatherv to correctly handle unequal local_num_documents
 * (remainder distribution gives some processes one extra document).
 *
 * @param data   Pointer to the local dataset partition
 * @param state  Pointer to the algorithm state (owns local assignments)
 * @param rank   MPI rank of the calling process
 * @param size   Total number of MPI processes
 */
static void gather_and_print_assignment(const Dataset *data,
                                        const AlgorithmState *state,
                                        int rank, int size) {
    int *local_doc_counts  = NULL;
    int *rank_offsets      = NULL;
    int *global_assignment = NULL;

    if (rank == 0) {
        local_doc_counts  = (int *)malloc(size * sizeof(int));
        rank_offsets      = (int *)calloc(size, sizeof(int));
        global_assignment = (int *)malloc(data->num_documents * sizeof(int));
    }

    /* Collect each rank's local document count into rank 0 */
    MPI_Gather(&data->local_num_documents, 1, MPI_INT,
               local_doc_counts,           1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int r = 1; r < size; r++)
            rank_offsets[r] = rank_offsets[r - 1] + local_doc_counts[r - 1];
    }

    MPI_Gatherv(state->assignment, data->local_num_documents, MPI_INT,
                global_assignment, local_doc_counts, rank_offsets, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_assignment(global_assignment, data->num_documents);

        free(global_assignment);
        free(local_doc_counts);
        free(rank_offsets);
    }
}
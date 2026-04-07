/**
 * Parallel and Distributed Computing 25/26
 * 
 * GROUP 33:
 * Artur Krystopchuk (104145) - arturkrystopchuk@tecnico.ulisboa.pt
 * João Martins (106819) - joao.bernardo.mota.martins@tecnico.ulisboa.pt
 * Sofia Ligeiro (116046) - sofiamligeiro@tecnico.ulisboa.pt
 *
 * Description: Handles input parsing and dataset partitioning. Calculates
 * the local document range for each MPI process and ensures each rank
 * only reads its assigned subset of data from the shared input file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parser-mpi.h"


#define SEED       1234
#define RAND_RANGE 10.0
#define UNIF01     ((double) rand() / RAND_MAX)


/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */

static Dataset *allocate_dataset(int num_cabinets, int num_documents,
                                 int num_subjects, int local_num_documents);
static int read_local_documents(FILE *file, const Dataset *data, int local_start);

static int generate_local_documents(const Dataset *data, int local_start);

/* =================================== *
 * PUBLIC FUNCTION DEFINITIONS         *
 * =================================== */

/**
 * Loads the subset of the dataset that belongs to this MPI process.
 * Every process opens the shared file independently and stores only
 * the documents in its assigned range [local_start, local_start + local_num_documents).
 *
 * @param filename  Path to the input file (accessible by all processes)
 * @param rank      MPI rank of the calling process
 * @param size      Total number of MPI processes
 * @return          Pointer to a newly allocated Dataset, or NULL on error
 */
Dataset *load_dataset(const char *filename, int rank, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    /* 1. Every process reads Header */
    int num_cabinets, num_documents, num_subjects;
    if (fscanf(file, "%d %d %d",
               &num_cabinets, &num_documents, &num_subjects)
            != NUMBER_EXPECTED_HEADER_ITEMS) {
        fprintf(stderr, "Rank %d: Failed to read header (C D S)\n", rank);
        fclose(file);
        return NULL;
    }

    /* 2. Determine this process's slice of the document range */
    int local_start, local_num_documents;
    compute_local_range(num_documents, rank, size, &local_start, &local_num_documents);

    /* 3. Allocate memory for the local partition */
    Dataset *data = allocate_dataset(num_cabinets, num_documents,
                                     num_subjects, local_num_documents);
    if (!data) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        fclose(file);
        return NULL;
    }
  
    /*
    4. Read all document lines; store only those that belong to this process 
    if (!read_local_documents(file, data, local_start)) {
        free_dataset(data);
        fclose(file);
        return NULL;
    }
    */

    /* 4. Generate subject scores for all documents */
    if (!generate_local_documents(data, local_start)) {
        free_dataset(data);
        return NULL;
    }



    fclose(file);
    return data;
}

/**
* Computes the local partition of documents for an MPI process.
 * Distributes documents across all processes as evenly as possible,
 * ensuring the difference in assigned documents is at most 1.
 *
 * @param D                   Total number of documents (global)
 * @param rank                MPI rank of the calling process
 * @param size                Total number of MPI processes
 * @param local_start         OUT: global index of the first document owned by this process
 * @param local_num_documents OUT: number of documents owned by this process
 */
void compute_local_range(int D, int rank, int size,
                         int *local_start, int *local_num_documents) {
    int base      = D / size;
    int remainder = D % size;

    *local_num_documents = base + (rank < remainder ? 1 : 0);
    *local_start = rank * base + (rank < remainder ? rank : remainder);
}

/**
 * Frees all memory associated with a Dataset struct.
 *
 * @param data Pointer to the dataset to free
 */
void free_dataset(Dataset *data) {
    if (!data) return;

    if (data->documents_scores) {
        free(data->documents_scores[0]); /* Frees the contiguous data block */
        free(data->documents_scores);    /* Frees the pointer array          */
    }

    free(data);
}

/* =================================== *
 * INTERNAL HELPER DEFINITIONS         *
 * =================================== */

/**
 * Allocates a Dataset struct and its local score matrix.
 * The score matrix is a single contiguous block exposed through a pointer array.
 * All scores are initialised to 0.
 *
 * @param num_cabinets         Global number of cabinets
 * @param num_documents        Global total number of documents
 * @param num_subjects         Number of subjects per document
 * @param local_num_documents  Number of documents owned by this process
 * @return                     Pointer to the allocated Dataset, or NULL on failure
 */
static Dataset *allocate_dataset(int num_cabinets, int num_documents,
                                 int num_subjects, int local_num_documents) {
    Dataset *data = (Dataset *)malloc(sizeof(Dataset));
    if (!data) return NULL;

    data->num_cabinets          = num_cabinets;
    data->num_documents         = num_documents;
    data->num_subjects          = num_subjects;
    data->local_num_documents   = local_num_documents;
    data->documents_scores      = NULL;

    /* 1. Allocate the contiguous data block: local_num_documents rows * num_subjects */
    double *block = (double *)calloc((size_t)local_num_documents * num_subjects,
                                     sizeof(double));
    if (!block) {
        free(data);
        return NULL;
    }

    /* 2. Allocate the row-pointer array */
    data->documents_scores = (double **)malloc(local_num_documents * sizeof(double *));
    if (!data->documents_scores) {
        free(block);
        free(data);
        return NULL;
    }

    /* 3. Wire each row pointer into the contiguous block */
    for (int d = 0; d < local_num_documents; d++)
        data->documents_scores[d] = &block[d * num_subjects];

    return data;
}

/**
 * Reads all D document entries from the open file stream, storing only
 * the documents whose IDs fall within [local_start, local_start + local_num_documents).
 * Out-of-range entries are read and discarded
 *
 * @param file        Open file stream positioned just after the header line
 * @param data        Pre-allocated Dataset
 * @param local_start Global index of the first document owned by this process
 * @return            TRUE on success, FALSE on any parse or bounds error
 */
static int read_local_documents(FILE *file, const Dataset *data, int local_start) {
    int    local_end = local_start + data->local_num_documents;
    int    S         = data->num_subjects;
    double score;

    for (int i = 0; i < data->num_documents; i++) {
        int doc_id;
        if (fscanf(file, "%d", &doc_id) != NUMBER_EXPECTED_SCALAR_ITEMS) {
            fprintf(stderr, "Error: Failed to read document ID at entry %d\n", i);
            return FALSE;
        }

        if (doc_id < 0 || doc_id >= data->num_documents) {
            fprintf(stderr, "Error: Document ID %d out of bounds [0, %d)\n",
                    doc_id, data->num_documents);
            return FALSE;
        }

        int is_local  = (doc_id >= local_start && doc_id < local_end);
        int local_idx = doc_id - local_start; /* Valid only when is_local */

        for (int s = 0; s < S; s++) {
            if (fscanf(file, "%lf", &score) != NUMBER_EXPECTED_SCALAR_ITEMS) {
                fprintf(stderr, "Error: Failed to read score for doc %d, subject %d\n",
                        doc_id, s);
                return FALSE;
            }
            if (is_local)
                data->documents_scores[local_idx][s] = score;
        }
    }

    return TRUE;
}

/**
 * Generates subject scores for the local document partition using a fixed seed.
 * All processes iterate over the full document range (size D) to maintain
 * the same global rand() sequence, ensuring that document i always receives
 * the same scores regardless of the number of MPI processes
 *
 * @param data         Pointer to the pre-allocated local dataset partition
 * @param local_start  Global index of the first document owned by this process
 * @return             TRUE on success, FALSE on any bounds error
 */
static int generate_local_documents(const Dataset *data, int local_start) {
    int local_end = local_start + data->local_num_documents;
    int S         = data->num_subjects;
    int D         = data->num_documents;

    if (local_start < 0 || local_end > data->num_documents) {
        fprintf(stderr, "Error: Local range [%d, %d) out of bounds [0, %d)\n",
                local_start, local_end, data->num_documents);
        return FALSE;
    }

    srand(SEED);

    for (int i = 0; i < D; i++) {
        int is_local  = (i >= local_start && i < local_end);
        int local_idx = i - local_start;

        for (int s = 0; s < S; s++) {
            double score = UNIF01 * RAND_RANGE;
            if (is_local)
                data->documents_scores[local_idx][s] = score;
        }
    }

    return TRUE;
}

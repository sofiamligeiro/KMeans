#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parser.h"

/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */
Dataset* allocate_dataset(int num_cabinets, int num_documents, int num_subjects);
int read_documents_data(FILE *file, Dataset *data);

/* =================================== *
 * PUBLIC FUNCTION DEFINITIONS         *
 * =================================== */

/**
 * Loads a new dataset struct from the input file
 *
 * @param filename String defining the input file name
 * @return new dataset struct
 */
Dataset* load_dataset(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    /* 1. Read Header */
    int num_cabinets, num_documents, num_subjects;
    if (fscanf(file, "%d %d %d", &num_cabinets, &num_documents, &num_subjects) != NUMBER_EXPECTED_HEADER_ITEMS) {
        fprintf(stderr, "Error: Failed to read header (C D S)\n");
        fclose(file);
        return NULL;
    }

    /* 2. Allocate Memory */
    Dataset *data = allocate_dataset(num_cabinets, num_documents, num_subjects);
    if (!data) {
        perror("Memory allocation failed for dataset");
        fclose(file);
        return NULL;
    }

    /* 3. Read Body Data */
    if (!read_documents_data(file, data)) {
        free_dataset(data); /* Assumes free_dataset properly frees the contiguous block and pointers */
        fclose(file);
        return NULL;
    }

    fclose(file);
    return data;
}

/**
 * Frees the memory allocated for a dataset struct
 *
 * @param data Pointer to the dataset
 */
void free_dataset(Dataset *data) {
    if (data == NULL) return;

    if (data->documents_scores != NULL) {
        free(data->documents_scores[0]); // Frees the data block
        free(data->documents_scores); // Frees the pointer array
    }

    free(data);
}

/**
 * Prints the information stored in a dataset (for debug purposes)
 *
 * @param data Pointer to the dataset
 */
void print_dataset(const Dataset *data) {
    if (data == NULL) {
        printf("Error: Cannot print, dataset is NULL.\n");
        return;
    }

    fprintf(stderr,"\n--- Dataset Overview ---\n");
    fprintf(stderr,"Cabinets (C): %d\n", data->num_cabinets);
    fprintf(stderr,"Documents (D): %d\n", data->num_documents);
    fprintf(stderr,"Subjects (S): %d\n", data->num_subjects);
    fprintf(stderr,"-------------------------\n");

    for (int document_id = 0; document_id < data->num_documents; document_id++) {
        fprintf(stderr,"Document ID %d:\tScores:", document_id);
        for (int subject_id = 0; subject_id < data->num_subjects; subject_id++) {
            fprintf(stderr," %.2f ", data->documents_scores[document_id][subject_id]);
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"-------------------------\n\n");
}

/* =================================== *
 * INTERNAL HELPER DEFINITIONS         *
 * =================================== */

/**
 * Allocates memory for a Dataset structure, including a contiguous,
 * SIMD-aligned 2D array for document scores. The columns (subjects)
 * are padded to ensure proper memory alignment for vectorized operations.
 *
 * @param num_cabinets  The total number of cabinets (stored in the dataset state).
 * @param num_documents The total number of documents (represents rows in the score matrix).
 * @param num_subjects  The total number of subjects per document (represents columns in the matrix).
 * @return              A pointer to the newly allocated Dataset struct, or NULL if memory allocation fails.
 */
Dataset* allocate_dataset(int num_cabinets, int num_documents, int num_subjects) {
    Dataset *data = (Dataset *)malloc(sizeof(Dataset));
    if (!data) return NULL;

    data->num_cabinets = num_cabinets;
    data->num_documents = num_documents;
    data->num_subjects = num_subjects;
    data->documents_scores = NULL;

    /* Calculate padded size (rounds up to nearest multiple of SIMD_PADDING_MULTIPLE) */
    int padded_num_subjects = ((num_subjects + (SIMD_PADDING_MULTIPLE - 1)) /
                               SIMD_PADDING_MULTIPLE) * SIMD_PADDING_MULTIPLE;

    size_t total_elements = (size_t)num_documents * padded_num_subjects;
    size_t total_bytes = total_elements * sizeof(double);

    /* 1. Allocate the contiguous data block */
    double *block_data = (double *)aligned_alloc(MEMORY_ALIGNMENT_BYTES, total_bytes);
    if (!block_data) {
        free(data);
        return NULL;
    }
    memset(block_data, 0, total_bytes);

    /* 2. Allocate the row pointers */
    data->documents_scores = (double **)malloc(num_documents * sizeof(double *));
    if (!data->documents_scores) {
        free(block_data);
        free(data);
        return NULL;
    }

    /* 3. Hook them up */
    for (int d = 0; d < num_documents; d++) {
        data->documents_scores[d] = &block_data[d * padded_num_subjects];
    }

    return data;
}

/**
 * Reads document score data from an open file stream and populates the Dataset structure.
 * It expects each entry in the file to start with a document ID, followed by exactly
 * 'num_subjects' floating-point scores.
 *
 * @param file A pointer to the opened file stream containing the document data.
 * @param data A pointer to the pre-allocated Dataset structure where the scores will be stored.
 * @return     TRUE if the data is parsed and stored successfully, FALSE if there was an error error
 * occurs or if an out-of-bounds document ID is read.
 */
int read_documents_data(FILE *file, Dataset *data) {
    int document_id;

    for (int i = 0; i < data->num_documents; i++) {
        /* Read Document ID */
        if (fscanf(file, "%d", &document_id) != NUMBER_EXPECTED_SCALAR_ITEMS) {
            fprintf(stderr, "Error: Failed to read document ID at index %d\n", i);
            return FALSE;
        }

        if (document_id < 0 || document_id >= data->num_documents) {
            fprintf(stderr, "Error: Document ID %d out of bounds\n", document_id);
            return FALSE;
        }

        /* Read Scores */
        for (int subject_id = 0; subject_id < data->num_subjects; subject_id++) {
            if (fscanf(file, "%lf", &data->documents_scores[document_id][subject_id]) != NUMBER_EXPECTED_SCALAR_ITEMS) {
                fprintf(stderr, "Error: Failed to read score for doc %d, subject %d\n", document_id, subject_id);
                return FALSE;
            }
        }
    }
    return TRUE;
}
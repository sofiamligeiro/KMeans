#include <stdio.h>
#include <stdlib.h>

#include "parser-omp.h"

/**
 * Loads a new dataset struct from the input file
 *
 * @param filename String defining the input file name
 * @return new dataset struct
 */
Dataset* load_dataset(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    Dataset *data = (Dataset *)malloc(sizeof(Dataset));
    if (data == NULL) {
        perror("Memory allocation failed for dataset");
        fclose(file);
        return NULL;
    }

    data->documents_scores = NULL;

    /* First line: C D S */
    if (fscanf(file, "%d %d %d", &data->num_cabinets, &data->num_documents, &data->num_subjects) != 3) {
        fprintf(stderr, "Error: Failed to read first line\n");
        free_dataset(data);
        fclose(file);
        return NULL;
    }

    /* 1. Allocate the contiguous data block */
    size_t total_elements = (size_t)data->num_documents * data->num_subjects;
    double *block_data = (double *)calloc(total_elements, sizeof(double));
    if (block_data == NULL) {
        free(data);
        fclose(file);
        return NULL;
    }

    /* 2. Allocate the row pointers */
    data->documents_scores = (double **)malloc(data->num_documents * sizeof(double *));
    if (data->documents_scores == NULL) {
        free(block_data);
        free(data);
        fclose(file);
        return NULL;
    }

    /* 3. Hook them up */
    for (int document_id = 0; document_id < data->num_documents; document_id++) {
        data->documents_scores[document_id] = &block_data[document_id * data->num_subjects];
    }

    /* Read the documents */
    for (int i = 0; i < data->num_documents; i++) {
        int docuement_id;
        if (fscanf(file, "%d", &docuement_id) != 1) {
            free_dataset(data);
            fclose(file);
            return NULL;
        }

        if (docuement_id < 0 || docuement_id >= data->num_documents) {
            free_dataset(data);
            fclose(file);
            return NULL;
        }

        /* Read directly using your new 2D syntax */
        for (int subject_id = 0; subject_id < data->num_subjects; subject_id++) {
            if (fscanf(file, "%lf", &data->documents_scores[docuement_id][subject_id]) != 1) {
                free_dataset(data);
                fclose(file);
                return NULL;
            }
        }
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
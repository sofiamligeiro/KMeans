#include <stdlib.h>
#include <stdio.h>

#include "algorithm.h"

/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */
static void compute_centroids(const Dataset *data, const int *doc_assignment, double **centroids);
static double compute_distance(const double *doc_scores, const double *centroid, int S);
static int reassign_docs(const Dataset *data, int *assignment, double **centroids);
static double** allocate_centroids(int C, int S);
static void free_centroids(double **centroids);

/* =================================== *
 * PUBLIC FUNCTION DEFINITIONS         *
 * =================================== */

/**
 * Distributes the documents to their respective cabinets
 *
 * @param data Pointer to the dataset
 * @return Array of the final cabinet assignments
 */
int* classify_docs(const Dataset *data) {
    int C = data->num_cabinets;
    int D = data->num_documents;
    int S = data->num_subjects;

    int *assignment = malloc(D * sizeof(int));
    if (!assignment) return NULL;

    for (int d = 0; d < D; d++) {    // Step 1 - round-robin initialization
        assignment[d] = d % C;
    }

    double **centroids = allocate_centroids(C, S);

    int moved = 1;
    while (moved) {
        compute_centroids(data, assignment, centroids);         // Step 2
        moved = reassign_docs(data, assignment, centroids);     // Step 3
    }

    free_centroids(centroids);

    return assignment;
}

/**
 * Print in stdout which cabinet each Document is assigned to
 *
 * @param assignment Array of the current cabinet assignments
 * @param D Number of documents
 * @return Pointer array to a matrix size C * S
 */
void print_assigment(int *assignment, int D) {
    for (int i = 0; i < D; i++) {
        fprintf(stdout, "%d\n", assignment[i]);
    }
}


/* =================================== *
 * INTERNAL HELPER DEFINITIONS         *
 * =================================== */

/**
 * Computes the centroids of the cabinets based on current document assignments
 *
 * @param data Pointer to the dataset
 * @param doc_assignment Array of current cabinet assignments for each document
 * @param centroids  Centroid matrix
 */
static void compute_centroids(const Dataset *data, const int *doc_assignment, double **centroids) {
    int C = data->num_cabinets;
    int S = data->num_subjects;
    int D = data->num_documents;

    int *docs_per_cabinet  = calloc(C, sizeof(int));
    if (!docs_per_cabinet) return;

    // centroid initialization to 0
    for (int c = 0; c < C; c++)
        for (int s = 0; s < S; s++)
            centroids[c][s] = 0.0;

    for (int d = 0; d < D; d++) {
        int c = doc_assignment[d];
        docs_per_cabinet[c]++;

        for (int s = 0; s < S; s++) {
            centroids[c][s] += data->documents_scores[d][s];
        }
    }

    for (int c = 0; c < C; c++) {
        if (docs_per_cabinet[c] > 0) {
            for (int s = 0; s < S; s++) {
                centroids[c][s] /= docs_per_cabinet[c];
            }
        }
    }

    free(docs_per_cabinet);
}

/**
 * Computes squared Euclidean distance between a document and a centroid
 *
 * @param doc_scores Array of the document's scores
 * @param centroid Array of centroid values
 * @param S Number of subjects
 * @return Squared Euclidean distance
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
 * Reassigns documents to the closest cabinet centroid
 *
 * @param data Pointer to the dataset
 * @param assignment Array of the current cabinet assignments
 * @param centroids Centroid matrix
 * @return 1 if any document changed cabinet, 0 otherwise
 */
static int reassign_docs(const Dataset *data, int *assignment, double **centroids) {
    int C = data->num_cabinets;
    int S = data->num_subjects;
    int D = data->num_documents;

    int moved = 0;

    for (int d = 0; d < D; d++) {
        int best_cabinet = assignment[d];

        double best_dist = compute_distance(data->documents_scores[d], centroids[best_cabinet], S);

        for (int c = 0; c < C; c++) {
            if (c != best_cabinet) {
                double dist = compute_distance(data->documents_scores[d], centroids[c], S);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cabinet = c;
                }
            }
        }
        if (best_cabinet != assignment[d]) {
            assignment[d] = best_cabinet;
            moved = 1;
        }
    }

    return moved;
}

/**
 * Efficient function to allocate large block of data
 * Important for cache
 *
 * @param C Number of cabinets
 * @param S Number of subjects
 * @return Pointer array to a matrix size C * S
 */
static double** allocate_centroids(int C, int S) {
    // 1. Allocate the contiguous data block
    double *block_data = calloc(C * S, sizeof(double));
    if (!block_data) return NULL;

    // 2. Allocate the row pointers
    double **centroids = malloc(C * sizeof(double*));
    if (!centroids) {
        free(block_data);
        return NULL;
    }

    // 3. Hook them up
    for (int i = 0; i < C; i++) {
        centroids[i] = &block_data[i * S];
    }

    return centroids;
}

/**
 * Free centroids
 *
 * @param centroids pointer array to a matrix size C * S
 */
static void free_centroids(double **centroids) {
    if (centroids) {
        free(centroids[0]); // Frees the data block
        free(centroids);    // Frees the pointer array
    }
}
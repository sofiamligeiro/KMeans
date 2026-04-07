#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>

#include "algorithm.h"

/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */
static void compute_centroids(const Dataset *data, const int *doc_assignment, double **centroids, int *num_docs_per_cabinet);
static double compute_distance(const double *doc_scores, const double *centroid, int S);
static int reassign_docs(const Dataset *data, int *assignment, double **centroids);
static double** allocate_centroids(int C, int S);
static void free_centroids(double **centroids);

static inline int round_up_to_simd_multiple(int num) {
    return (num + (SIMD_PADDING_MULTIPLE - 1)) & ~(SIMD_PADDING_MULTIPLE - 1);
}

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

    size_t assign_bytes = (D * sizeof(int) + (MEMORY_ALIGNMENT_BYTES - 1)) & ~(MEMORY_ALIGNMENT_BYTES - 1);
    int *assignment = (int *)aligned_alloc(MEMORY_ALIGNMENT_BYTES, assign_bytes);
    if (!assignment) return NULL;
    for (int d = 0; d < D; d++) {
        assignment[d] = d % C;
    }

    int *docs_per_cabinet = malloc(C * sizeof(int));
    if (!docs_per_cabinet) return NULL;

    double **centroids = allocate_centroids(C, S);

    int moved = 1;
    while (moved) {
        compute_centroids(data, assignment, centroids, docs_per_cabinet);
        moved = reassign_docs(data, assignment, centroids);
    }

    free_centroids(centroids);
    free(docs_per_cabinet);

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
 * @param centroids Centroid matrix
 * @param num_docs_per_cabinet Number of documents in each cabinet
 */
static void compute_centroids(const Dataset *data, const int *doc_assignment, double **centroids, int *num_docs_per_cabinet) {
    int C = data->num_cabinets;
    int S = data->num_subjects;
    int D = data->num_documents;
    int padded_S = round_up_to_simd_multiple(S);

    // Initialization to 0
    memset(num_docs_per_cabinet, 0, C * sizeof(int));
    for (int c = 0; c < C; c++) {
        memset(centroids[c], 0, padded_S * sizeof(double));
    }

    /* 2. Vectorized centroid accumulation */
    for (int d = 0; d < D; d++) {
        int current_cabinet_ID = doc_assignment[d];
        num_docs_per_cabinet[current_cabinet_ID]++;

        for (int s = 0; s < padded_S; s += SIMD_PADDING_MULTIPLE) {
            __m256d c_vec = _mm256_load_pd(&centroids[current_cabinet_ID][s]);
            __m256d d_vec = _mm256_load_pd(&data->documents_scores[d][s]);

            // centroids[c][s] += data->documents_scores[d][s];
            c_vec = _mm256_add_pd(c_vec, d_vec);
            _mm256_store_pd(&centroids[current_cabinet_ID][s], c_vec);
        }
    }

    /* 3. Vectorized mean calculation */
    for (int c = 0; c < C; c++) {
        if (num_docs_per_cabinet[c] > 0) {
            /* Optimization: Multiply by inverse instead of dividing inside the loop */
            double inv_num_docs_in_cab = 1.0 / num_docs_per_cabinet[c];
            __m256d inv_num_docs_in_cab_vec = _mm256_set1_pd(inv_num_docs_in_cab);

            for (int s = 0; s < padded_S; s += SIMD_PADDING_MULTIPLE) {
                __m256d c_vec = _mm256_load_pd(&centroids[c][s]);

                // centroids[c][s] /= num_docs_per_cabinet[c];
                c_vec = _mm256_mul_pd(c_vec, inv_num_docs_in_cab_vec);
                _mm256_store_pd(&centroids[c][s], c_vec);
            }
        }
    }
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
    __m256d sum_vec = _mm256_setzero_pd();
    int padded_S = round_up_to_simd_multiple(S);

    /* We safely iterate over padded_S because the tail pads are both 0.0 */
    for (int s = 0; s < padded_S; s += SIMD_PADDING_MULTIPLE) {
        __m256d d_vec = _mm256_load_pd(&doc_scores[s]);
        __m256d c_vec = _mm256_load_pd(&centroid[s]);

        // diff = doc_scores[s] - centroid[s];
        __m256d diff_vec = _mm256_sub_pd(d_vec, c_vec);

        // sum += diff * diff;
        sum_vec = _mm256_fmadd_pd(diff_vec, diff_vec, sum_vec);
    }

    /* sum_vec currently holds 4 partial sums: [ D3 | D2 | D1 | D0 ] */

    /* 1. Split the 256-bit register into two 128-bit registers (2 doubles each) */
    __m128d sum_vec_top_128 = _mm256_extractf128_pd(sum_vec, 1); // Extracts top half: [ D3 | D2 ]
    __m128d sum_vec_low_128 = _mm256_castpd256_pd128(sum_vec); // Extracts bottom half: [ D1 | D0 ]

    /* 2. Add the top half to the bottom half vertically */
    __m128d sum_128 = _mm_add_pd(sum_vec_top_128, sum_vec_low_128); // Now holds: [ D3+D1 | D2+D0 ]

    /* 3. Add the left side to the right side horizontally */
    __m128d final_sum = _mm_hadd_pd(sum_128, sum_128);  // Now holds: [ D3+D1+D2+D0 | D3+D1+D2+D0 ]

    /* 4. Extract the lowest 64-bit double from the register to return as a standard C double */
    return _mm_cvtsd_f64(final_sum);
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
    /* Calculate padded size (rounds up to nearest multiple of SIMD_PADDING_MULTIPLE) */
    int padded_S = round_up_to_simd_multiple(S);

    size_t total_elements = (size_t)C * padded_S;
    size_t total_bytes = total_elements * sizeof(double);

    /* 1. Allocate the contiguous data block with specific memory alignment */
    double *block_data = (double *)aligned_alloc(MEMORY_ALIGNMENT_BYTES, total_bytes);
    if (!block_data) return NULL;

    memset(block_data, 0, total_bytes);

    /* 2. Allocate the row pointers */
    double **centroids = (double **)malloc(C * sizeof(double*));
    if (!centroids) {
        free(block_data);
        return NULL;
    }

    /* 3. Hook them up using the padded size */
    for (int c = 0; c < C; c++) {
        centroids[c] = &block_data[c * padded_S];
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
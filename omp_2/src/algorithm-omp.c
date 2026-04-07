/**
 * Parallel and Distributed Computing 25/26
 * 
 * GROUP 33:
 * Artur Krystopchuk (104145) - arturkrystopchuk@tecnico.ulisboa.pt
 * João Martins (106819) - joao.bernardo.mota.martins@tecnico.ulisboa.pt
 * Sofia Ligeiro (116046) - sofiamligeiro@tecnico.ulisboa.pt
 */

 #include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "algorithm-omp.h"

/* =================================== *
 * INTERNAL HELPER FUNCTION PROTOTYPES *
 * =================================== */
static void compute_centroids(const Dataset *data, const int *assignment, 
                                       double *centroids, int *num_docs_per_cabinet,
                                       double *priv_centroids, int *num_docs_per_cabinet_priv);
static inline double compute_distance(const double *doc_scores, const double *centroid, int S);
static int reassign_docs(const Dataset *data, int *assignment, double **centroids);
static double **allocate_centroids(int C, int S);
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
int *classify_docs(const Dataset *data)
{
    int C = data->num_cabinets;
    int D = data->num_documents;
    int S = data->num_subjects;

    int *assignment = malloc(D * sizeof(int));
    int *num_docs_per_cabinet = malloc(C * sizeof(int));
    double **centroids = allocate_centroids(C, S);

    if (!assignment || !num_docs_per_cabinet || !centroids)
        return NULL;
    
    
    double *contiguous_block_centroids = centroids[0];
    int moved = TRUE;
    int local_moved;

    #pragma omp parallel 
    {
        #pragma omp for schedule(static)
        for (int d = 0; d < D; d++) // Step 1 - round-robin initialization
        {
            assignment[d] = d % C;
        }
        
        double *priv_centroids = calloc(C * S, sizeof(double));
        int *num_docs_per_cabinet_priv = calloc(C, sizeof(int));


        do {
            #pragma omp single
            {
                local_moved = FALSE;
                memset(num_docs_per_cabinet, 0, C * sizeof(int));
                memset(contiguous_block_centroids, 0, C * S * sizeof(double));
            }
            compute_centroids(data, assignment, contiguous_block_centroids, 
                                     num_docs_per_cabinet, priv_centroids, num_docs_per_cabinet_priv); // Step 2
            int thread_moved = reassign_docs(data, assignment, centroids);   // Step 3
            
            if (thread_moved)
            {
                local_moved = TRUE;
            }

            #pragma omp barrier
            #pragma omp single
            moved = local_moved;
            
        } while (moved);

        free(priv_centroids);
        free(num_docs_per_cabinet_priv);
    }

    free_centroids(centroids);
    free(num_docs_per_cabinet);
    return assignment;
}


/**
 * Print in stdout which cabinet each Document is assigned to
 *
 * @param assignment Array of the current cabinet assignments
 * @param D Number of documents
 * @return Pointer array to a matrix size C * S
 */
void print_assigment(int *assignment, int D)
{
    for (int i = 0; i < D; i++)
    {
        fprintf(stdout, "%d\n", assignment[i]);
    }
}


/* =================================== *
 * INTERNAL HELPER DEFINITIONS         *
 * =================================== */


/**
 * Computes cabinet centroids in parallel by accumulating values in thread-local 
 * buffers and then merging them into global arrays
 * 
 * Both centroids and priv_centroids are contiguous 
 *
 * @param data Pointer to the dataset
 * @param assignment Array with the current cabinet assignment of each document
 * @param centroids Contiguous memory block storing the centroids 
 * @param num_docs_per_cabinet Array with the number of documents per cabinet 
 * @param priv_centroids Thread-local centroid accumulators 
 * @param num_docs_per_cabinet_priv Thread-local document counters 
 */
static void compute_centroids(const Dataset *data, const int *assignment, 
                                       double *centroids, int *num_docs_per_cabinet,
                                       double *priv_centroids, int *num_docs_per_cabinet_priv)
{
    int C = data->num_cabinets;
    int S = data->num_subjects;
    int D = data->num_documents;

    memset(priv_centroids, 0, C * S * sizeof(double));
    memset(num_docs_per_cabinet_priv, 0, C * sizeof(int));

    
    #pragma omp for schedule(static)
    for (int d = 0; d < D; d++) {
        int c = assignment[d];
        num_docs_per_cabinet_priv[c]++;
  
        for (int s = 0; s < S; s++) {
            // NOTE: We can not do centroids[c][s] += data->documents_scores[d][s];
            //       bcz centroids[c] still points to the original contiguous block
            //       and not their local version
            priv_centroids[c * S + s] += data->documents_scores[d][s];
        }
    }

    #pragma omp critical
    {
        for (int i = 0; i < C; i++) {
            num_docs_per_cabinet[i] += num_docs_per_cabinet_priv[i];
        }     
        for (int i = 0; i < C * S; i++) {
            centroids[i] += priv_centroids[i];
        }
    }
    #pragma omp barrier

    
    #pragma omp for schedule(static)
    for (int c = 0; c < C; c++) {
        if (num_docs_per_cabinet[c] > 0) 
        {
            double inv_n = 1.0 / num_docs_per_cabinet[c];
            #pragma omp simd
            for (int s = 0; s < S; s++)
            {
                centroids[c * S + s] *= inv_n;
            } 
        }
    }
    #pragma omp barrier
}

/**
 * Computes squared Euclidean distance between a document and a centroid
 *
 * @param doc_scores Array of the document's scores
 * @param centroid Array of centroid values
 * @param S Number of subjects
 * @return Squared Euclidean distance
 */
static inline double compute_distance(const double *doc_scores, const double *centroid, int S)
{
    double dist = 0.0;

    #pragma omp simd reduction(+ : dist)
    for (int s = 0; s < S; s++)
    {
        double diff = doc_scores[s] - centroid[s];
        dist += diff * diff;
    }
    return dist;
}


/**
 * Reassigns documents to the closest cabinet centroid
 *
 * @param data Pointer to the dataset
 * @param assignment Array of the current cabinet assignments
 * @param centroids Centroid matrix
 * @return 1 if any document changed cabinet, 0 otherwise
 */
static int reassign_docs(const Dataset *data, int *assignment, double **centroids)
{
    int C = data->num_cabinets;
    int S = data->num_subjects;
    int D = data->num_documents;

    int thread_moved = FALSE;

    #pragma omp for schedule(static)
    for (int d = 0; d < D; d++) 
    {
        const double *doc_scores = data->documents_scores[d];
        int current_cabinet = assignment[d];
        double min_dist = compute_distance(doc_scores, centroids[current_cabinet], S);

        for (int c = 0; c < C; c++) 
        {
            if (c == current_cabinet) 
                continue;
            double dist = compute_distance(doc_scores, centroids[c], S);
            
            if (dist < min_dist) 
            {
                min_dist = dist;
                current_cabinet = c;
            }
        }

        if (current_cabinet != assignment[d]) 
        {
            assignment[d] = current_cabinet;
            thread_moved = TRUE;
        }
    }

    #pragma omp barrier 
    return thread_moved;
}

/**
 * Efficient function to allocate large block of data
 * Important for cache
 *
 * @param C Number of cabinets
 * @param S Number of subjects
 * @return Pointer array to a matrix size C * S
 */
static double **allocate_centroids(int C, int S)
{
    // 1. Allocate the contiguous data block

    double *block_data = calloc(C * S, sizeof(double));
    if (!block_data)
        return NULL;

    // 2. Allocate the row pointers
    double **centroids = malloc(C * sizeof(double *));
    if (!centroids)
    {
        free(block_data);
        return NULL;
    }

    // 3. Hook them up
    for (int i = 0; i < C; i++)
    {
        centroids[i] = &block_data[i * S];
    }

    return centroids;
}

/**
 * Free centroids
 *
 * @param centroids pointer array to a matrix size C * S
 */
static void free_centroids(double **centroids)
{
    if (centroids)
    {
        free(centroids[0]); // Frees the data block
        free(centroids);    // Frees the pointer array
    }
}

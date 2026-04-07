#ifndef DOCS_MPI_H
#define DOCS_MPI_H

#define TRUE  1
#define FALSE 0

typedef struct {
    int num_cabinets;           /* C - global number of cabinets              */
    int num_documents;          /* D - global number of documents             */
    int num_subjects;           /* S - global number of subjects per document */
    int local_num_documents;    /* Number of documents owned by this process  */
    double **documents_scores;  /* local_num_documents × S score matrix       */
} Dataset;

#endif /* DOCS_MPI_H */
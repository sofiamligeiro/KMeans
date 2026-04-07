#ifndef DOCS_H
#define DOCS_H

#define TRUE 1
#define FALSE 0

#define MEMORY_ALIGNMENT_BYTES 32
#define SIMD_PADDING_MULTIPLE 4

typedef struct {
    int num_cabinets;          /* C */
    int num_documents;         /* D */
    int num_subjects;          /* S */
    double **documents_scores; /* Array of size D * S */
} Dataset;

#endif /* DOCS_H */
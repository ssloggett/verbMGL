def align(seq1, seq2, S, insertion_penalty = -10, deletion_penalty = -10):
    """
    Find the optimum local sequence alignment for the sequences `seq1`
    and `seq2` using the Smith-Waterman algorithm. Optional keyword
    arguments give the gap-scoring scheme:

    `insertion_penalty` penalty for an insertion (default: -1)
    `deletion_penalty`  penalty for a deletion (default: -1)
    `S`  a matrix specifying the match score between elements
    """
    import numpy
    DELETION, INSERTION, MATCH = range(3)
    m, n = len(seq1), len(seq2)
    
    # Construct the similarity matrix in p[i][j], and remember how
    # it was constructed it -- insertion, deletion or (mis)match -- in
    # q[i][j]
    p = numpy.zeros((m + 1, n + 1))
    q = numpy.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            deletion = (p[i - 1][j] + deletion_penalty, DELETION)
            insertion = (p[i][j - 1] + insertion_penalty, INSERTION)
            match = (p[i - 1][j - 1] + S[seq1[i-1][1]][seq2[j-1][1]], MATCH)
            p[i][j], q[i][j] = max(deletion, insertion, match)
    # Yield the aligned sequences one character at a time in reverse order.
    def backtrack():
        i, j = m, n
        while i > 0 or j > 0:
            if i == 1:
                while j > 1:
                    j -= 1
                    yield ['*','*'], seq2[j]
                i,j=0,0
                yield seq1[i], seq2[j]
            elif j == 1:
                j = 0
                while i > 1:
                    i -= 1
                    yield seq1[i], ['*','*']
                i,j=0,0
                yield seq1[i], seq2[j]
            elif q[i][j] == MATCH:
                i -= 1
                j -= 1
                yield seq1[i], seq2[j]
            elif q[i][j] == INSERTION:
                j -= 1
                yield ['*','*'], seq2[j]
            elif q[i][j] == DELETION:
                i -= 1
                yield seq1[i], ['*','*']
    return [s[::-1] for s in zip(*backtrack())]
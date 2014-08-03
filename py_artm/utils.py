
def normalize(mat):
    """ Normalize columns so that each of them sum up to 1 """
    norms = mat.sum(0)
    norms[norms == 0] = 1
    return mat / norms
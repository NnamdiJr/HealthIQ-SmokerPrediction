__author__ = 'Nnamdi'

from scipy.sparse import vstack, hstack, csr_matrix


def appendMatrix(matrix01, matrix02):
    """Column-wise append matrix02 to matrix01, where IDs are unaligned. IDs are assumed to be vectors in first position
     of matrix data. Returns new matrix01."""
    userID_vector01 = matrix01[0] #Vector containing matrix01 userIDs
    userID_vector02 = matrix02[0] #Vector containing matrix02 userIDs
    hold_matrix = csr_matrix((csr_matrix.get_shape(matrix01)[0], csr_matrix.get_shape(matrix02)[1])) #Empty matrix

    for user01 in userID_vector01:
        for user02 in userID_vector02:
            if user01 == user02:
                hold_matrix = vstack([hold_matrix, matrix02[userID_vector02.index(user02),:]])
                break
            else:
                continue

    matrix01 = hstack([matrix01, hold_matrix])
    return matrix01
import argparse
import itertools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# This function takes the text file containing the matrix and returns a 2d numpy matrix. 
# The text file must contain a valid matrix with each row in a separate line like so : 
# 1100
# 1100
# 0011
# 0011
def text2numpy(text_file):

    # list to store rows as numpy arrays
    matrix = []
    with open(text_file) as f:
        for line in f:
            # Read each line and append
            row = list(line.strip())
            matrix.append(np.array(row, dtype=np.uint8))
        # Check if matrix has a valid shape
        if len(set([row.shape[0] for row in matrix])) > 1:
            raise ValueError("File must contain valid binary matrix")
        # Convert to 2d numpy array and return
        matrix = np.array(matrix)
        return matrix
    

def compare_fragments(frag1, frag2):

    # First check if they are directly identical
    if frag1.shape == frag2.shape and np.array_equal(frag1, frag2): 
        return True
    
    # Rotation
    frag2_r = np.rot90(frag2)   # counter clockwise 90
    if frag1.shape == frag2_r.shape and np.array_equal(frag1, frag2_r):
        return True
    frag2_r = np.rot90(frag2, 3)    # clockwise 90
    if frag1.shape == frag2_r.shape and np.array_equal(frag1, frag2_r):
        return True
    frag2_r = np.rot90(frag2, 2)    # rotate both 90 -> rotate one by 180
    if frag1.shape == frag2_r.shape and np.array_equal(frag1, frag2_r):
        return True
    
    # Flip
    # Flipping does not change shape
    # Flipping along 0 axis, then 1 (or vice versa) is same as 180 rotation
    if frag1.shape == frag2.shape:
        frag1_f0 = np.flip(frag1, 0)    # flip along axis=1
        frag1_f1 = np.flip(frag1, 1)    # flip along axis=1
        frag2_f0 = np.flip(frag2, 0)    # flip along axis=0
        frag2_f1 = np.flip(frag2, 1)    # flip along axis=1

        if np.array_equal(frag1_f0, frag2_f0) or np.array_equal(frag1_f1, frag2_f0):
            return True
        if np.array_equal(frag1_f0, frag2_f1) or np.array_equal(frag1_f1, frag2_f1):
            return True
    
    return False


def find_fragments(matrix_file):

    # Convert text in file to 2d numpy array
    matrix = text2numpy(matrix_file)

    # Now we need to iterate over all possible valid fragments
    # and check whether they repeat in the matrix.
    # Fragments must be at least 2 pixels wide/tall and may be rotated 90 degrees, shifted or flipped.
    min_fragment_len = 2

    # Store num rows and columns
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]

    # We need to iterate over row and col dimensions.
    # We add 1 to the values since shape (dimension) values start from 1 and not 0
    row_range = [l for l in list(range(n_rows+1)) if l >= min_fragment_len]
    col_range = [l for l in list(range(n_cols+1)) if l >= min_fragment_len]
    frag_shapes = list(itertools.product(row_range, col_range))

    # Remove permutations since we need to compare nxm fragements with mxn fragements as well
    # Ex : [(2,3), (3,2), (3,3)] -> [(2,3), (4,4)]
    frag_shapes = [tuple(j) if len(j) > 1 else tuple((next(iter(j)), next(iter(j)))) 
                   for j in set([frozenset(i) for i in frag_shapes])]

    # Get rid of the full matrix shape.
    sorted_frag_shapes = sorted(frag_shapes, key=lambda x: x[0]+x[1])[:-1]
    
    # To store output fragments
    output_fragments_by_shape = {}

    # Get all possible fragments of each shape and its transpose.
    # Compare fragments with each other.
    # Store valid fragments.
    for frag_size in sorted_frag_shapes:

        # Store all fragments of frag_size
        valid_fragments = []

        # Sliding_window gives frag_size sized blocks from the matrix
        fragments = sliding_window_view(matrix, frag_size)
        # Resize to get an array of fragments
        fragments = fragments.reshape(-1, *frag_size)
        for f in range(fragments.shape[0]):
            valid_fragments.append(fragments[f])

        # If not a square fragment, transpose and slide window
        if frag_size[0] != frag_size[1]:
            frag_size_t = tuple(reversed(frag_size))

            # Need to check whether transpose is valid shape within matrix
            if frag_size_t[0] <= n_rows and frag_size_t[1] <= n_cols:
                fragments = sliding_window_view(matrix, frag_size_t)
                fragments = fragments.reshape(-1, *frag_size_t)
                for f in range(fragments.shape[0]):
                    valid_fragments.append(fragments[f])

        # Compare all pairs : get list indices
        id_pairs = list(itertools.combinations(list(range(len(valid_fragments))), 2))

        # To check whether a repeating fragment is already considered
        repeat_ids = set()
        # To store a minimal repeating set
        repetitions = set()

        # Compare fragments under rotation and flip
        for id_pair in id_pairs:
            # If both fragments already considered then skip.
            # Else if at least one fragment is not seen, need to check if it repeats with a seen one.
            # Otherwise check if two unseen fragments are repititions.
            if id_pair[0] not in repeat_ids or id_pair[1] not in repeat_ids:
                if compare_fragments(valid_fragments[id_pair[0]], valid_fragments[id_pair[1]]):
                    # If new fragment, need to store only one of the fragments for final output
                    if id_pair[0] not in repeat_ids and id_pair[1] not in repeat_ids:
                        repetitions.add(id_pair[0]) 
                    # Store ids to check repetition
                    repeat_ids.add(id_pair[0])
                    repeat_ids.add(id_pair[1])

        # Get fragments from their ids
        output_fragments = [valid_fragments[r]for r in list(repetitions)]
        # Store final output fragments by size
        output_fragments_by_shape[frag_size] = output_fragments

    return output_fragments_by_shape


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--matrix_file",
        default=None,
        type=str,
        help="Text file that contains matrix, one row per line. Please make sure file contains only 1s and 0s without space"
    )

    # parse args
    args = parser.parse_args()
    # main function for finding fragments
    output_fragments_by_shape = find_fragments(args.matrix_file)
    # print recovered fragments
    for key, val in output_fragments_by_shape.items():
        print('shape: {}'.format(key))
        for frag in val:
            print(frag)
            print('')

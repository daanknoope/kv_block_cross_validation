name = 'kv_block_cross_validation'
from typing import List, NewType

Indices = NewType('Indices', List[int])


def create_train_test_ranges(n_rows: int, h: int, v: int) -> object:
    """
    Creates training and testing sets for dependent (i.e. temporal) datasets using hv-block cross validation,
    as described in: "Consistent cross-validatory model-selection for dependent data: hv-block cross-validation",
    by Jeff Racine, 2000, https://www.sciencedirect.com/science/article/pii/S0304407600000300
    :param n_rows: number of rows in dataframe
    :param h: maximum lag
    :param v: fold size
    """
    train_sets = []
    test_sets = []

    n_folds = int(n_rows / v)

    if n_folds < 1:
        raise ValueError(f"Cannot create less than 1 fold (with {n_rows} rows and {v} elements per fold)")

    for fold in range(n_folds):
        fold_start = fold * v
        fold_end = (fold + 1) * v
        test_sets.append([*range(fold_start, fold_end)])

        train_set = []

        # add left side of test
        if fold_start - h > 0:  # otherwise padding is bigger than training size
            train_set += [*range(0, fold_start - h)]

        # add right side of test
        if fold_end - h > 0:
            train_set += [*range(fold_end + h, n_rows)]

        train_sets.append(train_set)

    return train_sets, test_sets

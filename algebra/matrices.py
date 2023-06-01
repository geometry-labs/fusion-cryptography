from copy import deepcopy
from typing import List, Union


def is_algebraic_class(cls):
    required_methods = {"__eq__", "__add__", "__neg__", "__sub__", "__mul__"}
    return all(hasattr(cls, method) for method in required_methods)


class GeneralMatrix:
    elem_class: type
    matrix: List[List[object]]

    def __init__(self, matrix: List[list]):
        if not isinstance(matrix, list):
            raise ValueError("Matrix must be a list")
        elif not matrix:
            raise ValueError("Matrix must not be empty.")
        elif any(not isinstance(x, list) for x in matrix):
            raise ValueError("Matrix must be a list of lists")
        elif any(not x for x in matrix):
            raise ValueError("Matrix must not contain empty lists")
        elif not all(len(row) == len(matrix[0]) for row in matrix):
            raise ValueError("All rows must have the same length")
        elif not is_algebraic_class(cls=matrix[0][0].__class__):
            raise ValueError(
                "Matrix must contain only instances of the same algebraic class"
            )
        elif not all(
            isinstance(row, list)
            and all(isinstance(item, matrix[0][0].__class__) for item in row)
            for row in matrix
        ):
            raise ValueError(
                "Matrix must contain only instances of the same algebraic class"
            )
        self.elem_class = matrix[0][0].__class__
        self.matrix = matrix

    def __str__(self):
        return f"GeneralMatrix(elem_class={self.elem_class}, matrix={self.matrix})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.matrix)

    def __iter__(self):
        return iter(self.matrix)

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __delitem__(self, key):
        self.matrix[key] = 0

    def __eq__(self, other):
        if other == 0:
            return all(all(item == 0 for item in row) for row in self.matrix)
        elif (
            not isinstance(other, GeneralMatrix) or self.elem_class != other.elem_class
        ):
            return False
        elif len(self.matrix) != len(other.matrix) or len(self.matrix[0]) != len(
            other.matrix[0]
        ):
            return False
        return self.matrix == other.matrix

    def __add__(self, other):
        if other == 0:
            return self
        elif (
            not isinstance(other, GeneralMatrix) or self.elem_class != other.elem_class
        ):
            raise NotImplementedError(
                "Can only add GeneralMatrix objects of the same algebraic class"
            )
        elif len(self.matrix) != len(other.matrix) or len(self.matrix[0]) != len(
            other.matrix[0]
        ):
            raise ValueError("Matrix dimensions must match")
        resulting_matrix = [
            [self.matrix[i][j] + other.matrix[i][j] for j in range(len(self.matrix[0]))]
            for i in range(len(self.matrix))
        ]
        return GeneralMatrix(matrix=resulting_matrix)

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __neg__(self):
        resulting_matrix = [
            [-self.matrix[i][j] for j in range(len(self.matrix[0]))]
            for i in range(len(self.matrix))
        ]
        return GeneralMatrix(matrix=resulting_matrix)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, self.elem_class):
            resulting_matrix = [
                [self.matrix[i][j] * other for j in range(len(self.matrix[0]))]
                for i in range(len(self.matrix))
            ]
            return GeneralMatrix(matrix=resulting_matrix)
        elif (
            not isinstance(other, GeneralMatrix) or self.elem_class != other.elem_class
        ):
            raise TypeError("Can only multiply matrices of the same algebraic class")
        elif len(self.matrix[0]) != len(other.matrix):
            raise ValueError("Matrix dimension mismatch")
        result: GeneralMatrix = deepcopy(self)
        result.matrix = [
            [0 for j in range(len(other.matrix[0]))] for i in range(len(self.matrix))
        ]
        for i in range(len(self.matrix)):
            for j in range(len(other.matrix[0])):
                next_data = self.matrix[i][0] * other.matrix[0][j]
                for k in range(1, len(self.matrix[0])):
                    next_data += self.matrix[i][k] * other.matrix[k][j]
                result.matrix[i][j] = next_data
        return result

    def __mod__(self, other):
        if not isinstance(other, int):
            raise TypeError("Can only take the remainder of a matrix with an integer")
        elif other <= 1:
            raise ValueError("Modulus must be greater than 1")
        resulting_matrix = [
            [self.matrix[i][j] % other for j in range(len(self.matrix[0]))]
            for i in range(len(self.matrix))
        ]
        return GeneralMatrix(matrix=resulting_matrix)

    def norm(self, p: Union[int, str]):
        if not all(hasattr(z, "norm") for y in self.matrix for z in y):
            raise NotImplementedError("Matrix elements must have a norm method")
        elif p == "infty":
            return max(max(z.norm(p=p) for z in y) for y in self.matrix)

    def weight(self):
        if not all(hasattr(z, "weight") for y in self.matrix for z in y):
            raise NotImplementedError("Matrix elements must have a weight method")
        return max(max(z.weight() for z in y) for y in self.matrix)

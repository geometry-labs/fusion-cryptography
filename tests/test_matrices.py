import pytest
from random import randrange

from algebra.matrices import *
from algebra.ntt import find_primitive_root
from algebra.polynomials import (
    PolynomialCoefficientRepresentation as Poly,
    PolynomialNTTRepresentation as PolyNTT,
    transform,
)
from tests.test_ntt import TEST_2D_Q_PAIRS, TEST_SAMPLE_SIZE


def test_is_algebraic_class():
    assert not is_algebraic_class(cls="hello world".__class__())
    assert is_algebraic_class(Poly)
    assert is_algebraic_class(PolyNTT)


def test_general_matrix():
    for degree, modulus in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=modulus, root_order=2 * degree)
        inv_root: int = pow(root, modulus - 2, modulus)
        for _ in range(TEST_SAMPLE_SIZE):
            # Generate random left-matrix [[a_left, b_left], [c_left, d_left]]
            a_left_coef: int = randrange(1, modulus)
            a_left_index: int = randrange(degree)
            a_left_coefs: List[int] = [
                0 if i != a_left_index else a_left_coef for i in range(degree)
            ]
            a_left: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=a_left_coefs,
            )

            b_left_coef: int = randrange(1, modulus)
            b_left_index: int = randrange(degree)
            b_left_coefs: List[int] = [
                0 if i != b_left_index else b_left_coef for i in range(degree)
            ]
            b_left: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=b_left_coefs,
            )

            c_left_coef: int = randrange(1, modulus)
            c_left_index: int = randrange(degree)
            c_left_coefs: List[int] = [
                0 if i != c_left_index else c_left_coef for i in range(degree)
            ]
            c_left: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=c_left_coefs,
            )

            d_left_coef: int = randrange(1, modulus)
            d_left_index: int = randrange(degree)
            d_left_coefs: List[int] = [
                0 if i != d_left_index else d_left_coef for i in range(degree)
            ]
            d_left: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=d_left_coefs,
            )

            a_left_hat: PolyNTT = transform(x=deepcopy(a_left))
            b_left_hat: PolyNTT = transform(x=deepcopy(b_left))
            c_left_hat: PolyNTT = transform(x=deepcopy(c_left))
            d_left_hat: PolyNTT = transform(
                x=deepcopy(d_left)
            )  # not to be confused with degree d

            left_matrix: GeneralMatrix = GeneralMatrix(
                matrix=[
                    [deepcopy(a_left), deepcopy(b_left)],
                    [deepcopy(c_left), deepcopy(d_left)],
                ]
            )
            left_matrix_hat: GeneralMatrix = GeneralMatrix(
                matrix=[
                    [deepcopy(a_left_hat), deepcopy(b_left_hat)],
                    [deepcopy(c_left_hat), deepcopy(d_left_hat)],
                ]
            )

            # Test the left-matrix
            assert left_matrix.matrix[0][0] == a_left
            assert left_matrix.matrix[0][1] == b_left
            assert left_matrix.matrix[1][0] == c_left
            assert left_matrix.matrix[1][1] == d_left
            assert left_matrix.elem_class == Poly

            # Test the left_matrix_hat
            assert left_matrix_hat.matrix[0][0] == a_left_hat
            assert left_matrix_hat.matrix[0][1] == b_left_hat
            assert left_matrix_hat.matrix[1][0] == c_left_hat
            assert left_matrix_hat.matrix[1][1] == d_left_hat
            assert left_matrix_hat.elem_class == PolyNTT

            # Generate random right-matrix [[a_left, b_left], [c_left, d_left]]
            a_right_coef: int = randrange(1, modulus)
            a_right_index: int = randrange(degree)
            a_right_coefs: List[int] = [
                0 if i != a_right_index else a_right_coef for i in range(degree)
            ]
            a_right: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=a_right_coefs,
            )

            b_right_coef: int = randrange(1, modulus)
            b_right_index: int = randrange(degree)
            b_right_coefs: List[int] = [
                0 if i != b_right_index else b_right_coef for i in range(degree)
            ]
            b_right: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=b_right_coefs,
            )

            c_right_coef: int = randrange(1, modulus)
            c_right_index: int = randrange(degree)
            c_right_coefs: List[int] = [
                0 if i != c_right_index else c_right_coef for i in range(degree)
            ]
            c_right: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=c_right_coefs,
            )

            d_right_coef: int = randrange(1, modulus)
            d_right_index: int = randrange(degree)
            d_right_coefs: List[int] = [
                0 if i != d_right_index else d_right_coef for i in range(degree)
            ]
            d_right: Poly = Poly(
                modulus=modulus,
                degree=degree,
                root_order=2 * degree,
                root=root,
                inv_root=inv_root,
                coefficients=d_right_coefs,
            )

            a_right_hat: PolyNTT = transform(x=deepcopy(a_right))
            b_right_hat: PolyNTT = transform(x=deepcopy(b_right))
            c_right_hat: PolyNTT = transform(x=deepcopy(c_right))
            d_right_hat: PolyNTT = transform(
                x=deepcopy(d_right)
            )  # not to be confused with degree d

            right_matrix: GeneralMatrix = GeneralMatrix(
                matrix=[
                    [deepcopy(a_right), deepcopy(b_right)],
                    [deepcopy(c_right), deepcopy(d_right)],
                ]
            )
            right_matrix_hat: GeneralMatrix = GeneralMatrix(
                matrix=[
                    [deepcopy(a_right_hat), deepcopy(b_right_hat)],
                    [deepcopy(c_right_hat), deepcopy(d_right_hat)],
                ]
            )

            # Test the right-matrix
            assert right_matrix.matrix[0][0] == a_right
            assert right_matrix.matrix[0][1] == b_right
            assert right_matrix.matrix[1][0] == c_right
            assert right_matrix.matrix[1][1] == d_right
            assert right_matrix.elem_class == Poly

            # Test the right_matrix_hat
            assert right_matrix_hat.matrix[0][0] == a_right_hat
            assert right_matrix_hat.matrix[0][1] == b_right_hat
            assert right_matrix_hat.matrix[1][0] == c_right_hat
            assert right_matrix_hat.matrix[1][1] == d_right_hat
            assert right_matrix_hat.elem_class == PolyNTT

            # Test the left-matrix * right-matrix
            expected_product: GeneralMatrix = GeneralMatrix(
                matrix=[
                    [
                        a_left * a_right + b_left * c_right,
                        a_left * b_right + b_left * d_right,
                    ],
                    [
                        c_left * a_right + d_left * c_right,
                        c_left * b_right + d_left * d_right,
                    ],
                ]
            )
            observed_product: GeneralMatrix = left_matrix * right_matrix
            assert observed_product == expected_product

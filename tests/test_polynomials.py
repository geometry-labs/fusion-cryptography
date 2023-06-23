import pytest
from copy import deepcopy
from random import randrange
from typing import List

import pytest

from algebra.ntt import find_primitive_root, ntt_poly_mult
from algebra.polynomials import (
    PolynomialCoefficientRepresentation as Poly,
    PolynomialNTTRepresentation as PolyNTT,
    transform,
    sample_polynomial_coefficient_representation,
)
from test_ntt import TEST_2D_Q_PAIRS, TEST_SAMPLE_SIZE


def test_arithmetic():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(q) for _ in range(d)]
            b_coefs: List[int] = [randrange(q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=b_coefs,
            )
            expected_c_coefs: List[int] = [0 for _ in range(2 * d)]
            # foil
            for i, next_a in enumerate(a_coefs):
                for j, next_b in enumerate(b_coefs):
                    expected_c_coefs[i + j] = (
                        expected_c_coefs[i + j] + next_a * next_b
                    ) % q
            expected_c_coefs = [
                (x - y) % q for x, y in zip(expected_c_coefs[:d], expected_c_coefs[d:])
            ]
            expected_c: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=expected_c_coefs,
            )

            observed_c: Poly = a * b
            assert len(observed_c.coefficients) == len(expected_c.coefficients)
            assert all(
                (x - y) % q == 0
                for x, y in zip(observed_c.coefficients, expected_c.coefficients)
            )

            another_observed_c: List[int] = ntt_poly_mult(
                f=deepcopy(a_coefs),
                g=deepcopy(b_coefs),
                modulus=q,
                root=root,
                inv_root=inv_root,
                root_order=2 * d,
            )
            right: List[int] = [-y for y in another_observed_c[d:]]
            another_observed_c = another_observed_c[:d]
            while len(right) >= d:
                next_right = right[:d]
                right = [-y for y in right[d:]]
                for i, y in enumerate(next_right):
                    another_observed_c[i] = (another_observed_c[i] + y) % q
            for i, y in enumerate(right):
                another_observed_c[i] = (another_observed_c[i] + y) % q
            assert len(another_observed_c) == len(expected_c.coefficients)
            assert all(
                (x - y) % q == 0
                for x, y in zip(observed_c.coefficients, another_observed_c)
            )
            assert all(
                (x - y) % q == 0
                for x, y in zip(expected_c.coefficients, observed_c.coefficients)
            )
            try:
                assert expected_c == observed_c
            except AssertionError:
                print()
                assert expected_c == observed_c
            # Since expected_c and observed_c are equivalent, we only need one in the rest of this test.

            a_hat: PolyNTT = transform(x=a)
            b_hat: PolyNTT = transform(x=b)
            c_hat: PolyNTT = transform(x=observed_c)
            a_hat_times_b_hat: PolyNTT = a_hat * b_hat
            inv_a_hat: Poly = transform(x=a_hat)
            inv_b_hat: Poly = transform(x=b_hat)
            inv_a_hat_times_b_hat: Poly = transform(x=a_hat_times_b_hat)
            inv_c_hat: Poly = transform(c_hat)
            assert inv_a_hat == a
            assert inv_b_hat == b
            assert inv_a_hat_times_b_hat == observed_c == inv_c_hat


def test_monomial_products():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coef: int = randrange(1, q)
            a_index: int = randrange(d)
            a_coefs: List[int] = [0 if i != a_index else a_coef for i in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )

            b_coef: int = randrange(1, q)
            b_index: int = randrange(d)
            b_coefs: List[int] = [0 if i != b_index else b_coef for i in range(d)]
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=b_coefs,
            )

            expected_c_coefs: List[int] = [0 for _ in range(d)]
            expected_c_coefs[(a_index + b_index) % d] = ((a_coef * b_coef) % q) * (
                1 - 2 * int(a_index + b_index >= d)
            )
            expected_c: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=expected_c_coefs,
            )
            observed_c: Poly = a * b
            try:
                assert expected_c == observed_c
            except AssertionError:
                print()
                expected_c: Poly = Poly(
                    modulus=q,
                    degree=d,
                    root_order=2 * d,
                    root=root,
                    inv_root=inv_root,
                    coefficients=expected_c_coefs,
                )
                observed_c: Poly = a * b
                assert expected_c == observed_c

            a_hat: PolyNTT = transform(x=a)
            b_hat: PolyNTT = transform(x=b)
            a_hat_times_b_hat: PolyNTT = a_hat * b_hat
            inv_a_hat_times_b_hat: Poly = transform(x=a_hat_times_b_hat)
            assert inv_a_hat_times_b_hat == observed_c


# Test PolynomialCoefficientRepresentation.__init__
def test_poly_init():
    with pytest.raises(ValueError):
        Poly(modulus=1, degree=1, root_order=1, root=1, inv_root=1, coefficients=1)
    with pytest.raises(TypeError):
        Poly(modulus=5, degree=2, root_order=1, root=1, inv_root=1, coefficients=1)
    with pytest.raises(TypeError):
        Poly(
            modulus=1.0,
            degree=1,
            root_order=1,
            root=1,
            inv_root=1,
            coefficients=["hello world"],
        )
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            assert a.modulus == q
            assert a.degree == d
            assert a.root_order == 2 * d
            assert a.root == root
            assert a.inv_root == inv_root
            assert a.coefficients == a_coefs


def test_poly_str():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            assert (
                str(a)
                == f"PolynomialCoefficientRepresentation(modulus={q}, degree={d}, root={root}, inv_root={inv_root}, root_order={2 * d}, coefficients={a_coefs})"
            )


# Test PolynomialCoefficientRepresentation.__repr__
def test_poly_repr():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            assert (
                repr(a)
                == f"PolynomialCoefficientRepresentation(modulus={q}, degree={d}, root={root}, inv_root={inv_root}, root_order={2 * d}, coefficients={a_coefs})"
            )


# Test PolynomialCoefficientRepresentation.__eq__
def test_poly_eq():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a.coefficients,
            )
            assert a == a
            assert a == b
            assert len(a.coefficients) == len(b.coefficients)
            for i, next_coef in enumerate(a.coefficients):
                a.coefficients[i] = next_coef + randrange(2**10) * q
                b.coefficients[i] = next_coef + randrange(2**10) * q
            assert a == b


# Test PolynomialCoefficientRepresentation.__add__
def test_poly_add():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            b_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=b_coefs,
            )
            c: Poly = a + b
            assert c.modulus == q
            assert c.degree == d
            assert c.root_order == 2 * d
            assert c.root == root
            assert c.inv_root == inv_root
            for i in range(d):
                assert (
                    c.coefficients[i] - (a.coefficients[i] + b.coefficients[i])
                ) % q == 0


# Test PolynomialCoefficientRepresentation.__sub__
def test_poly_sub():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            b_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=b_coefs,
            )
            c: Poly = a - b
            assert c.modulus == q
            assert c.degree == d
            assert c.root_order == 2 * d
            assert c.root == root
            assert c.inv_root == inv_root
            for i in range(d):
                assert (
                    c.coefficients[i] - (a.coefficients[i] - b.coefficients[i])
                ) % q == 0


# Test PolynomialCoefficientRepresentation.__mul__
def test_poly_mul():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            b_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            expected_c_coefs: List[int] = [0 for _ in range(2 * d)]
            for i, next_a_coef in enumerate(a_coefs):
                for j, next_b_coef in enumerate(b_coefs):
                    expected_c_coefs[i + j] = (
                        expected_c_coefs[i + j] + (next_a_coef * next_b_coef)
                    ) % q
            expected_c_coefs = [
                (x - y) % q for x, y in zip(expected_c_coefs[:d], expected_c_coefs[d:])
            ]

            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=b_coefs,
            )
            expected_c: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=expected_c_coefs,
            )
            observed_c: Poly = a * b
            assert expected_c == observed_c


# Test PolynomialCoefficientRepresentation.norm
def test_poly_norm():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            expected_infinity_norm: int = max(abs(x) for x in a.coefficients)
            expected_one_norm: int = sum(abs(x) for x in a.coefficients)
            expected_two_norm: float = sum(abs(x) ** 2 for x in a.coefficients) ** 0.5
            observed_infinity_norm: int = a.norm(p="infty")
            assert expected_infinity_norm == observed_infinity_norm
            with pytest.raises(NotImplementedError):
                observed_one_norm = a.norm(p=1)
            with pytest.raises(NotImplementedError):
                observed_two_norm = a.norm(p=2)


# Test PolynomialNTTRepresentation.__init__
def test_poly_ntt_init():
    with pytest.raises(ValueError):
        PolyNTT(modulus=2, degree=1, root_order=2, root=1, inv_root=1, values=1)
    with pytest.raises(TypeError):
        PolyNTT(modulus=5, degree=2, root_order=2, root=-1, inv_root=-1, values=1)
    with pytest.raises(ValueError):
        PolyNTT(
            modulus=5,
            degree=2,
            root_order=2,
            root=1,
            inv_root=1,
            values=["hello world"],
        )
    root = find_primitive_root(modulus=5, root_order=2)
    inv_root = pow(root, 5 - 2, 5)
    with pytest.raises(TypeError):
        PolyNTT(
            modulus=5,
            degree=2,
            root_order=2,
            root=root,
            inv_root=inv_root,
            values=["hello world"],
        )
    with pytest.raises(ValueError):
        PolyNTT(
            modulus=5, degree=2, root_order=2, root=root, inv_root=inv_root, values=[1]
        )
    with pytest.raises(ValueError):
        PolyNTT(
            modulus=5,
            degree=2,
            root_order=2,
            root=root,
            inv_root=inv_root,
            values=[1, 2, 3],
        )
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a_coefs,
            )
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                coefficients=a.coefficients,
            )
            assert a == b
            assert a == a
            for i, next_coef in enumerate(a.coefficients):
                a.coefficients[i] = next_coef + randrange(2) * q
                b.coefficients[i] = next_coef + randrange(2) * q
            assert a == b


# Test PolynomialNTTRepresentation.__str__
def test_poly_ntt_str():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            assert (
                str(a_hat)
                == repr(a_hat)
                == f"PolynomialNTTRepresentation(modulus={q}, degree={d}, root={root}, inv_root={inv_root}, root_order={2 * d}, values={a_vals})"
            )


# Test PolynomialNTTRepresentation.__eq__
def test_poly_ntt_eq():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            b_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_hat.values,
            )
            assert a_hat == b_hat
            assert a_hat == a_hat
            for i, next_val in enumerate(a_hat.values):
                a_hat.values[i] = next_val + randrange(2) * q
                b_hat.values[i] = next_val + randrange(2) * q
            assert a_hat == b_hat


# Test PolynomialNTTRepresentation.__add__
def test_poly_ntt_add():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            b_vals: List[int] = [randrange(1, q) for _ in range(d)]
            b_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=b_vals,
            )
            c_hat: PolyNTT = a_hat + b_hat
            assert all(
                [
                    (z - (x + y)) % q == 0
                    for x, y, z in zip(a_vals, b_vals, c_hat.values)
                ]
            )
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_hat + b_vals
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_vals + b_hat
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_hat + b_vals[0]
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_vals[0] + b_hat
            c_hat: PolyNTT = a_hat + 0
            assert all((x - y) % q == 0 for x, y in zip(a_vals, c_hat.values))
            c_hat: PolyNTT = 0 + b_hat
            assert all((x - y) % q == 0 for x, y in zip(b_vals, c_hat.values))
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_hat + q


# Test PolynomialNTTRepresentation.__sub__
def test_poly_ntt_sub():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            b_vals: List[int] = [randrange(1, q) for _ in range(d)]
            b_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=b_vals,
            )
            c_hat: PolyNTT = a_hat - b_hat
            for i in range(d):
                assert (c_hat.values[i] - (a_vals[i] - b_vals[i])) % q == 0
            with pytest.raises(TypeError):
                c_hat: PolyNTT = a_hat - b_vals
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_vals - b_hat
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_hat - b_vals[0]
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_vals[0] - b_hat
            c_hat: PolyNTT = a_hat - 0
            assert c_hat == a_hat

            assert b_hat - 0 == b_hat + 0 == b_hat
            assert 0 - b_hat == -(b_hat - 0) == -b_hat

            assert 1 * b_hat == b_hat * 1
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = a_hat - q


# Test PolynomialNTTRepresentation.__neg__
def test_poly_ntt_neg():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            b_hat: PolyNTT = -a_hat
            for i in range(d):
                assert (b_hat.values[i] + a_vals[i]) % q == 0


# Test PolynomialNTTRepresentation.__radd__
def test_poly_ntt_radd():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            b_vals: List[int] = [randrange(1, q) for _ in range(d)]
            b_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=b_vals,
            )
            c_hat: PolyNTT = b_hat + a_hat
            for i in range(d):
                assert (c_hat.values[i] - (a_vals[i] + b_vals[i])) % q == 0
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = b_hat + a_vals
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = b_vals + a_hat
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = b_hat + a_vals[0]
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = b_vals[0] + a_hat
            c_hat: PolyNTT = b_hat + 0
            assert c_hat == b_hat
            c_hat: PolyNTT = 0 + a_hat
            assert c_hat == a_hat
            with pytest.raises(NotImplementedError):
                c_hat: PolyNTT = b_hat + q


# Test PolynomialNTTRepresentation.__mul__
def test_poly_ntt_mul():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            b_vals: List[int] = [randrange(1, q) for _ in range(d)]
            b_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=b_vals,
            )
            c_hat: PolyNTT = a_hat * b_hat
            for i in range(d):
                assert (c_hat.values[i] - (a_vals[i] * b_vals[i])) % q == 0
            with pytest.raises(NotImplementedError):
                a_hat * b_vals
            with pytest.raises(NotImplementedError):
                a_vals * b_hat

            c_hat: PolyNTT = a_hat * 0
            assert c_hat == 0
            c_hat: int = 0 * b_hat
            assert c_hat == 0
            c_hat: PolyNTT = a_hat * 1
            assert c_hat == a_hat
            c_hat: PolyNTT = 1 * b_hat
            assert c_hat == b_hat


# Test PolynomialNTTRepresentation.__rmul__
def test_poly_ntt_rmul():
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_vals: List[int] = [randrange(1, q) for _ in range(d)]
            a_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=a_vals,
            )
            b_vals: List[int] = [randrange(1, q) for _ in range(d)]
            b_hat: PolyNTT = PolyNTT(
                modulus=q,
                degree=d,
                root_order=2 * d,
                root=root,
                inv_root=inv_root,
                values=b_vals,
            )
            c_hat: PolyNTT = b_hat * a_hat
            for i in range(d):
                assert (c_hat.values[i] - (a_vals[i] * b_vals[i])) % q == 0
            with pytest.raises(NotImplementedError):
                b_hat * a_vals
            with pytest.raises(NotImplementedError):
                b_vals * a_hat

            c_hat: PolyNTT = b_hat * 0
            assert c_hat == 0
            c_hat: int = 0 * a_hat
            assert c_hat == 0
            c_hat: PolyNTT = b_hat * 1
            assert c_hat == b_hat
            c_hat: PolyNTT = 1 * a_hat
            assert c_hat == a_hat


# Test transform
def test_transform_2d():
    with pytest.raises(NotImplementedError):
        transform(x="hello, world!")
    for d, q in TEST_2D_Q_PAIRS:
        root: int = find_primitive_root(modulus=q, root_order=2 * d)
        inv_root: int = pow(root, q - 2, q)
        for _ in range(TEST_SAMPLE_SIZE):
            a_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root=root,
                inv_root=inv_root,
                root_order=2 * d,
                coefficients=a_coefs,
            )
            a_hat: PolyNTT = transform(x=a)

            b_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root=root,
                inv_root=inv_root,
                root_order=2 * d,
                coefficients=b_coefs,
            )
            b_hat: PolyNTT = transform(x=b)

            c_coefs: List[int] = [0 for _ in range(2 * d)]
            for i in range(len(a_coefs)):
                for j in range(len(b_coefs)):
                    c_coefs[i + j] = (c_coefs[i + j] + a_coefs[i] * b_coefs[j]) % q
            c_coefs = [(x - y) % q for x, y in zip(c_coefs[:d], c_coefs[d:])]
            c: Poly = Poly(
                modulus=q,
                degree=d,
                root=root,
                inv_root=inv_root,
                root_order=2 * d,
                coefficients=c_coefs,
            )

            a_ntt: PolyNTT = transform(x=a)
            b_ntt: PolyNTT = transform(x=b)
            a_ntt_times_b_ntt: PolyNTT = a_ntt * b_ntt
            inv_a_ntt_times_b_ntt: Poly = transform(x=a_ntt_times_b_ntt)
            assert inv_a_ntt_times_b_ntt == c


def test_comprehensive():
    for _ in range(TEST_SAMPLE_SIZE):
        for d, q in TEST_2D_Q_PAIRS:
            root: int = find_primitive_root(modulus=q, root_order=2 * d)
            inv_root: int = pow(root, q - 2, q)
            # Random polynomial a
            a_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            a: Poly = Poly(
                modulus=q,
                degree=d,
                root=root,
                inv_root=inv_root,
                root_order=2 * d,
                coefficients=a_coefs,
            )
            # Random polynomial b
            b_coefs: List[int] = [randrange(1, q) for _ in range(d)]
            b: Poly = Poly(
                modulus=q,
                degree=d,
                root=root,
                inv_root=inv_root,
                root_order=2 * d,
                coefficients=b_coefs,
            )

            # Transformed a and b
            a_hat: PolyNTT = transform(x=a)
            b_hat: PolyNTT = transform(x=b)

            # Product of transforms
            a_hat_times_b_hat: PolyNTT = a_hat * b_hat

            # Invert
            inv_a_hat_times_b_hat: Poly = transform(x=a_hat_times_b_hat)

            # Check
            assert inv_a_hat_times_b_hat == a * b


# test sample_polynomial_coefficient_representation
def test_sample_polynomial_coefficient_representation():
    modulus: int = 65537
    degree: int = 1024
    root_order: int = 2 * degree
    root: int = find_primitive_root(modulus=modulus, root_order=root_order)
    inv_root: int = pow(root, modulus - 2, modulus)
    norm_bound: int = 1000
    weight_bound: int = 100
    seed: int = 123456789
    f: Poly = sample_polynomial_coefficient_representation(
        modulus=modulus,
        degree=degree,
        root=root,
        inv_root=inv_root,
        root_order=root_order,
        norm_bound=norm_bound,
        weight_bound=weight_bound,
        seed=seed,
    )
    assert isinstance(f, Poly)
    assert f.modulus == modulus
    assert f.degree == degree
    assert f.root == root
    assert f.inv_root == inv_root
    assert f.root_order == root_order
    assert len(f.coefficients) == degree
    assert f.norm(p="infty") <= norm_bound
    assert f.weight() <= weight_bound

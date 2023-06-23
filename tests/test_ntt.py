import pytest
from copy import deepcopy
from random import randrange
from typing import List, Tuple

from algebra.ntt import (
    is_odd_prime,
    find_primitive_root,
    has_primitive_root_of_unity,
    is_primitive_root,
    ntt_poly_mult,
    cooley_tukey_ntt,
    gentleman_sande_intt,
    bit_reverse_copy,
    cent,
)

TEST_SAMPLE_SIZE: int = 2**5
LOG2_D_MIN: int = 2
LOG2_D_MAX: int = 6
Q_MAX: int = 2**17
TEST_2D_Q_PAIRS: List[Tuple[int, int]] = []
TEST_D_Q_PAIRS: List[Tuple[int, int]] = []
for log2d in range(LOG2_D_MIN, LOG2_D_MAX + 1):
    d: int = 1 << log2d
    q: int = 2 * d + 1
    while q < Q_MAX:
        while not is_odd_prime(q) and q < Q_MAX:
            q += 2 * d
        if is_odd_prime(q) and q < Q_MAX:
            TEST_2D_Q_PAIRS.append((d, q))
            find_primitive_root(
                modulus=q, root_order=2 * d
            )  # precompute roots of unity.
            q *= 2
            q -= (q - 1) % (2 * d)
            assert (q - 1) % (2 * d) == 0
for log2d in range(LOG2_D_MIN, LOG2_D_MAX + 1):
    d: int = 1 << log2d
    q: int = d + 1
    while q < Q_MAX:
        while not is_odd_prime(q) and q < Q_MAX:
            q += d
        if is_odd_prime(q) and q < Q_MAX:
            TEST_D_Q_PAIRS.append((d, q))
            find_primitive_root(modulus=q, root_order=d)  # precompute roots of unity.
            q *= 2
            q -= (q - 1) % d
            assert (q - 1) % d == 0


def test_inverse():
    for d, q in TEST_2D_Q_PAIRS:
        x = randrange(1, q)
        assert pow(x, q - 1, q) == 1
        y = pow(x, q - 2, q)
        assert (x * y) % q == 1


def test_ntt_poly_mult_scalars():
    touched_d = dict()
    for d, q in TEST_2D_Q_PAIRS:
        if (
            d not in touched_d
        ):  # just test the first prime for each choice of d for now.
            touched_d[d] = True
            # Make some parameters
            root_order: int = 2 * d
            assert has_primitive_root_of_unity(modulus=q, root_order=root_order)
            r: int = find_primitive_root(modulus=q, root_order=root_order)
            assert is_primitive_root(val=r, modulus=q, root_order=root_order)
            inv_r: int = pow(r, q - 2, q)
            assert (r * inv_r) % q == 1

            # sample a random polynomial f
            f: List[int] = [randrange(q) for _ in range(d)]
            assert len(f) == d
            # sample a random scalar g, encoded as a degree-0 polynomial
            g: List[int] = [randrange(q)] + [0 for _ in range(d - 1)]
            assert len(g) == d

            expected_h_by_inspection: List[int] = [(next_f * g[0]) % q for next_f in f]
            assert len(expected_h_by_inspection) == d

            expected_h_by_foil: List[int] = [0 for _ in range(2 * d)]
            for i, next_f in enumerate(f):
                for j, next_g in enumerate(g):
                    expected_h_by_foil[i + j] += (next_f * next_g) % q
            expected_h_by_foil = [
                (x - y) % q
                for x, y in zip(expected_h_by_foil[:d], expected_h_by_foil[d:])
            ]
            assert len(expected_h_by_foil) == d
            try:
                assert len(expected_h_by_inspection) == len(expected_h_by_foil) == d
            except AssertionError:
                print()
                assert len(expected_h_by_inspection) == len(expected_h_by_foil) == d
            for x, y in zip(expected_h_by_inspection, expected_h_by_foil):
                try:
                    assert (x - y) % q == 0
                except AssertionError:
                    print()
                    assert (x - y) % q == 0
            expected_h = expected_h_by_foil
            observed_h: List[int] = ntt_poly_mult(
                f=f, g=g, modulus=q, root=r, inv_root=inv_r, root_order=root_order
            )
            try:
                assert len(observed_h) == len(expected_h) == d
            except AssertionError:
                print()
                assert len(observed_h) == len(expected_h) == d
            try:
                assert all((x - y) % q == 0 for x, y in zip(expected_h, observed_h))
            except AssertionError:
                print()
                assert all((x - y) % q == 0 for x, y in zip(expected_h, observed_h))


def test_ntt_poly_mult_monomials():
    # test that X**i * X**j = X**(i+j) mod (X**d + 1) for all i, j in range(d)
    touched_d = dict()
    for d, q in TEST_2D_Q_PAIRS:
        if d not in touched_d:
            touched_d[d] = True
            # Make some parameters
            root_order: int = 2 * d
            assert has_primitive_root_of_unity(modulus=q, root_order=root_order)
            r: int = find_primitive_root(modulus=q, root_order=root_order)
            assert is_primitive_root(val=r, modulus=q, root_order=root_order)
            inv_r: int = pow(r, q - 2, q)
            assert (r * inv_r) % q == 1
            for non_zero_coefficient_in_f in range(d):
                f: List[int] = [0 for _ in range(d)]
                f[non_zero_coefficient_in_f] = 1
                for non_zero_coefficient_in_g in range(d):
                    g: List[int] = [0 for _ in range(d)]
                    g[non_zero_coefficient_in_g] = 1

                    expected_h: List[int] = [0 for _ in range(2 * d)]
                    expected_h[
                        non_zero_coefficient_in_f + non_zero_coefficient_in_g
                    ] = 1
                    # reduce
                    expected_h = [
                        (x - y) % q for x, y in zip(expected_h[:d], expected_h[d:])
                    ]
                    try:
                        assert all(
                            expected_h[i] % q == 0
                            for i in range(d)
                            if i
                            != (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                            % d
                        )
                    except AssertionError:
                        print()
                        assert all(
                            expected_h[i] % q == 0
                            for i in range(d)
                            if i
                            != (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                            % d
                        )
                    if non_zero_coefficient_in_f + non_zero_coefficient_in_g >= d:
                        assert (
                            expected_h[
                                (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                                % d
                            ]
                            + 1
                        ) % q == 0
                    else:
                        assert (
                            expected_h[
                                (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                                % d
                            ]
                            - 1
                        ) % q == 0
                    observed_h: List[int] = ntt_poly_mult(
                        f=f,
                        g=g,
                        modulus=q,
                        root=r,
                        inv_root=inv_r,
                        root_order=root_order,
                    )
                    try:
                        assert len(observed_h) == len(expected_h) == d
                    except AssertionError:
                        print()
                        assert len(observed_h) == len(expected_h) == d
                    try:
                        assert all(
                            (x - y) % q == 0 for x, y in zip(expected_h, observed_h)
                        )
                    except AssertionError:
                        print()
                        assert all(
                            (x - y) % q == 0 for x, y in zip(expected_h, observed_h)
                        )


def test_ntt_poly_mult_scalars_with_monomials():
    # test that aX**i * bX**j = (ab)X**(i+j) mod (q, X**d + 1) for all i, j in range(d)
    touched_d = dict()
    for d, q in TEST_2D_Q_PAIRS:
        if d not in touched_d:
            touched_d[d] = True
            # Make some parameters
            root_order: int = 2 * d
            assert has_primitive_root_of_unity(modulus=q, root_order=root_order)
            r: int = find_primitive_root(modulus=q, root_order=root_order)
            assert is_primitive_root(val=r, modulus=q, root_order=root_order)
            inv_r: int = pow(r, q - 2, q)
            assert (r * inv_r) % q == 1
            for non_zero_coefficient_in_f in range(d):
                f: List[int] = [0 for _ in range(d)]
                f[non_zero_coefficient_in_f] = randrange(1, q)
                for non_zero_coefficient_in_g in range(d):
                    g: List[int] = [0 for _ in range(d)]
                    g[non_zero_coefficient_in_g] = randrange(1, q)

                    expected_h: List[int] = [0 for _ in range(2 * d)]
                    expected_h[
                        non_zero_coefficient_in_f + non_zero_coefficient_in_g
                    ] = (f[non_zero_coefficient_in_f] * g[non_zero_coefficient_in_g])
                    # reduce
                    expected_h = [
                        (x - y) % q for x, y in zip(expected_h[:d], expected_h[d:])
                    ]
                    try:
                        assert all(
                            expected_h[i] % q == 0
                            for i in range(d)
                            if i
                            != (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                            % d
                        )
                    except AssertionError:
                        print()
                        assert all(
                            expected_h[i] % q == 0
                            for i in range(d)
                            if i
                            != (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                            % d
                        )
                    if non_zero_coefficient_in_f + non_zero_coefficient_in_g >= d:
                        assert (
                            expected_h[
                                (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                                % d
                            ]
                            + (
                                (
                                    f[non_zero_coefficient_in_f]
                                    * g[non_zero_coefficient_in_g]
                                )
                                % q
                            )
                        ) % q == 0
                    else:
                        assert (
                            expected_h[
                                (non_zero_coefficient_in_f + non_zero_coefficient_in_g)
                                % d
                            ]
                            - (
                                (
                                    f[non_zero_coefficient_in_f]
                                    * g[non_zero_coefficient_in_g]
                                )
                                % q
                            )
                        ) % q == 0
                    observed_h: List[int] = ntt_poly_mult(
                        f=f,
                        g=g,
                        modulus=q,
                        root=r,
                        inv_root=inv_r,
                        root_order=root_order,
                    )
                    try:
                        assert len(observed_h) == len(expected_h) == d
                    except AssertionError:
                        print()
                        assert len(observed_h) == len(expected_h) == d
                    try:
                        assert all(
                            (x - y) % q == 0 for x, y in zip(expected_h, observed_h)
                        )
                    except AssertionError:
                        print()
                        assert all(
                            (x - y) % q == 0 for x, y in zip(expected_h, observed_h)
                        )


def test_poly_mult_simple():
    modulus = 17
    halfmod = 8
    logmod = 5
    root_order = 16
    degree_bound = 8
    root = find_primitive_root(modulus=modulus, root_order=root_order)
    root_powers = [root**i for i in range(degree_bound)]
    inv_root = pow(root, modulus - 2, modulus)
    inv_root_powers = [inv_root**i for i in range(degree_bound)]
    assert all((x * y) % modulus == 1 for x, y in zip(root_powers, inv_root_powers))
    bit_rev_root_powers = bit_reverse_copy(root_powers)
    bit_rev_inv_root_powers = bit_reverse_copy(inv_root_powers)

    f_coefs = [0, 1, 0, 0, 0, 0, 0, 0]
    cooley_tukey_ntt(
        val=f_coefs,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )  # in place transformation
    ntt_of_f_coefs = deepcopy(f_coefs)
    gentleman_sande_intt(
        val=f_coefs,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )  # in place transformation

    g_coefs = [1, 2, 3, 4, 5, 6, 7, 8]
    cooley_tukey_ntt(
        val=g_coefs,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )  # in place transformation
    ntt_of_g_coefs = deepcopy(g_coefs)
    gentleman_sande_intt(
        val=g_coefs,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )  # in place transformation

    fg_coefs_from_ntt_poly_mult = ntt_poly_mult(
        f=f_coefs,
        g=g_coefs,
        modulus=modulus,
        root=root,
        inv_root=inv_root,
        root_order=root_order,
    )

    h_from_foiling = [0 for _ in range(2 * degree_bound)]
    for i, f_coef in enumerate(f_coefs):
        for j, g_coef in enumerate(g_coefs):
            h_from_foiling[i + j] += f_coef * g_coef
    h_from_foiling = [
        cent(val=x - y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for x, y in zip(h_from_foiling[:degree_bound], h_from_foiling[degree_bound:])
    ]

    another_equivalent_form_of_h = [-8, 1, 2, 3, 4, 5, 6, 7]
    assert len(h_from_foiling) == len(another_equivalent_form_of_h) == degree_bound
    for x, y in zip(h_from_foiling, another_equivalent_form_of_h):
        assert (x - y) % modulus == 0  # equivalent?

    # expected cooley_tukey_ntt of h_from_foiling
    cooley_tukey_ntt(
        val=h_from_foiling,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )  # in-place transfomration
    ntt_of_h_from_foiling = deepcopy(h_from_foiling)
    # transform h_from_foiling back
    gentleman_sande_intt(
        val=h_from_foiling,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )  # in-place transformation

    # Now check that component-wise product of ntt_of_f_coefs and ntt_of_g_coefs is equivalent to ntt_of_h_from_foiling
    component_wise_product_of_ntt_of_f_and_ntt_of_g = [
        cent(val=x * y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for x, y in zip(ntt_of_f_coefs, ntt_of_g_coefs)
    ]
    for x, y in zip(
        component_wise_product_of_ntt_of_f_and_ntt_of_g, ntt_of_h_from_foiling
    ):
        assert (x - y) % modulus == 0  # equivalent?


def test_ntt_poly_mult_basic():
    for d, q in TEST_2D_Q_PAIRS:
        # Make some parameters
        root_order: int = 2 * d
        assert has_primitive_root_of_unity(modulus=q, root_order=root_order)
        r: int = find_primitive_root(modulus=q, root_order=root_order)
        assert is_primitive_root(val=r, modulus=q, root_order=root_order)
        inv_r: int = pow(r, q - 2, q)
        assert (r * inv_r) % q == 1

        for _ in range(TEST_SAMPLE_SIZE):
            # pick two random polynomials with degree bound d
            f: List[int] = [randrange(q) for _ in range(d)]
            g: List[int] = [randrange(q) for _ in range(d)]

            # Constructed expected f(X)*g(X) by foiling then reducing mod X**d + 1
            expected_h: List[int] = [0 for _ in range(2 * d)]
            # foil
            for i, next_f in enumerate(f):
                for j, next_g in enumerate(g):
                    expected_h[i + j] += next_f * next_g
            # reduce
            expected_h: List[int] = [
                (x - y) % q for x, y in zip(expected_h[:d], expected_h[d:])
            ]
            assert len(expected_h) == d

            # NTT-multiply-INTT
            observed_h: List[int] = ntt_poly_mult(
                f=f, g=g, modulus=q, root=r, inv_root=inv_r, root_order=root_order
            )
            assert len(observed_h) == d
            assert all((x - y) % q == 0 for x, y in zip(expected_h, observed_h))


def test_ntt_poly_mult_against_one():
    # Make some parameters
    q: int = 5
    d: int = 2
    root_order: int = 2 * d
    assert has_primitive_root_of_unity(modulus=q, root_order=root_order)
    r: int = find_primitive_root(modulus=q, root_order=root_order)
    assert is_primitive_root(val=r, modulus=q, root_order=root_order)
    inv_r: int = pow(r, q - 2, q)
    assert (r * inv_r) % q == 1

    for _ in range(TEST_SAMPLE_SIZE):
        # pick two random polynomials with degree bound d
        one: List[int] = [1] + [0 for _ in range(d - 1)]
        g: List[int] = [randrange(q) for _ in range(d)]

        # Constructed expected f(X)*g(X) by foiling then reducing mod X**d + 1
        expected_h: List[int] = [0 for _ in range(2 * d)]
        # foil
        for i, next_f in enumerate(one):
            for j, next_g in enumerate(g):
                expected_h[i + j] += next_f * next_g
        # reduce
        expected_h: List[int] = [
            (x - y) % q for x, y in zip(expected_h[:d], expected_h[d:])
        ]
        assert len(expected_h) == d

        # NTT-multiply-INTT
        observed_h: List[int] = ntt_poly_mult(
            f=one, g=g, modulus=q, root=r, inv_root=inv_r, root_order=root_order
        )
        assert len(observed_h) == d
        assert all((x - y) % q == 0 for x, y in zip(expected_h, observed_h))

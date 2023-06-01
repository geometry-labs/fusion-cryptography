"""
The ntt module handles the Number Theoretic Transform (NTT) and its inverse in constant time.
"""
from copy import deepcopy
from math import ceil, log2
from typing import Dict, Tuple, List, Optional

CACHED_PRIMITIVE_ROOTS: Dict[Tuple[int, int], int] = {}
CACHED_IS_ODD_PRIME: Dict[int, bool] = {}
CACHED_HAS_PRIMITIVE_ROOT_OF_UNITY: Dict[Tuple[int, int], bool] = {}
CACHED_IS_POW_TWO_GEQ_TWO: Dict[int, bool] = {}
CACHED_IS_ROOT_OF_UNITY: Dict[Tuple[int, int, int], bool] = {}
CACHED_IS_PRIMITIVE_ROOT_OF_UNITY: Dict[Tuple[int, int, int], bool] = {}
CACHED_FIND_PRIMITIVE_ROOT: Dict[Tuple[int, int], Optional[int]] = {}


def is_odd_prime(val: int) -> bool:
    """
    Check if x is an odd prime number.
    :param val: Input number.
    :type val: int
    :return b: Boolean indicating whether x is an odd prime.
    :rtype: bool
    """
    if val not in CACHED_IS_ODD_PRIME:
        CACHED_IS_ODD_PRIME[val] = False
        if (
            isinstance(val, int)
            and val >= 3
            and all(val % i != 0 for i in range(3, int(val**0.5) + 1, 2))
        ):
            CACHED_IS_ODD_PRIME[val] = True
    return CACHED_IS_ODD_PRIME[val]


def has_primitive_root_of_unity(modulus: int, root_order: int) -> bool:
    """
    Check if a modulus has a primitive root of unity of a given order.
    :param modulus: Integer modulus
    :type modulus: int
    :param root_order: Order of primitive root of unity
    :type root_order: int
    :return: True if modulus has a primitive root of unity of order root_order, False otherwise
    """
    if (modulus, root_order) not in CACHED_HAS_PRIMITIVE_ROOT_OF_UNITY:
        CACHED_HAS_PRIMITIVE_ROOT_OF_UNITY[(modulus, root_order)] = False
        if (
            isinstance(modulus, int)
            and isinstance(root_order, int)
            and modulus >= 3
            and root_order >= 2
        ):
            CACHED_HAS_PRIMITIVE_ROOT_OF_UNITY[(modulus, root_order)] = (
                modulus - 1
            ) % root_order == 0
    return CACHED_HAS_PRIMITIVE_ROOT_OF_UNITY[(modulus, root_order)]


def is_pow_two_geq_two(val: int) -> bool:
    """
    Check if a number is a power of two.
    :param val: Input integer
    :type val: int
    :return: True if val is a power of two, False otherwise
    :rtype: bool
    """
    if val not in CACHED_IS_POW_TWO_GEQ_TWO:
        CACHED_IS_POW_TWO_GEQ_TWO[val] = False
        if isinstance(val, int) and val >= 2:
            CACHED_IS_POW_TWO_GEQ_TWO[val] = (val & (val - 1)) == 0
    return CACHED_IS_POW_TWO_GEQ_TWO[val]


def bit_reverse_copy(val: list):
    """
    Permute indices by bit-reversal.
    :param val: Input list of anything
    :type val: list
    :return lav: Output list matches input list with indices bit-reversed
    :rtype: list
    """
    if not isinstance(val, list):
        raise ValueError("Input must be a list")
    n: int = len(val)
    k: int = n.bit_length() - 1
    bit_reversed_indices: List[int] = [
        int(bin(i)[2:].zfill(k)[::-1], 2) for i in range(n)
    ]
    lav = [deepcopy(val[i]) for i in bit_reversed_indices]  # bit reverse
    return lav


def cent(val: int, modulus: int, halfmod: int, logmod: int) -> int:
    """
    Centrally reduce a value modulo q in constant time. Output val satisfies -(q//2) <= val <= q//2.
    :param modulus: Input modulus
    :type modulus: int
    :param halfmod: Half the modulus
    :type halfmod: int
    :param logmod: bits to describe integers with this modulus
    :type logmod: int
    :param val: Input integer
    :type val: int
    :return z: such that z = val % q and -(q//2) <= z <= q//2
    :rtype: int
    """
    if (
        not isinstance(val, int)
        or not isinstance(modulus, int)
        or not isinstance(halfmod, int)
        or not isinstance(logmod, int)
    ):
        raise TypeError("Input must be integers")
    elif modulus < 2:
        raise ValueError("Modulus must be at least 2")
    elif halfmod < 1:
        raise ValueError("Halfmod must be at least 1")
    elif logmod < 1:
        raise ValueError("Logmod must be at least 1")
    y: int = val % modulus
    intermediate_value: int = y - halfmod - 1
    z: int = y - (1 + (intermediate_value >> logmod)) * modulus
    return z


def is_root_of_unity(val: int, modulus: int, root_order: int) -> bool:
    """
    Check if val is a root of unity of order root_order modulo modulus.
    :param val: Input integer
    :type val: int
    :param modulus: Input modulus
    :type modulus: int
    :param root_order: Order of the root of unity
    :type root_order: int
    :return b: Boolean indicating whether val is a root of unity of order root_order modulo modulus
    :rtype: bool
    """
    if (val, modulus, root_order) not in CACHED_IS_ROOT_OF_UNITY:
        CACHED_IS_ROOT_OF_UNITY[(val, modulus, root_order)] = False
        if (
            isinstance(val, int)
            and isinstance(modulus, int)
            and isinstance(root_order, int)
            and modulus >= 2
            and root_order >= 1
        ):
            CACHED_IS_ROOT_OF_UNITY[(val, modulus, root_order)] = (
                pow(val, root_order, modulus) == 1
            )
    return CACHED_IS_ROOT_OF_UNITY[(val, modulus, root_order)]


def is_primitive_root(val: int, modulus: int, root_order: int) -> bool:
    """
    Check if val is a primitive root of order root_order modulo modulus.
    :param val: Input integer
    :type val: int
    :param modulus: Input modulus
    :type modulus: int
    :param root_order: Order of the root of unity
    :type root_order: int
    :return b: Boolean indicating whether val is a primitive root of order root_order modulo modulus
    :rtype: bool
    """
    if (val, modulus, root_order) not in CACHED_IS_PRIMITIVE_ROOT_OF_UNITY:
        CACHED_IS_PRIMITIVE_ROOT_OF_UNITY[(val, modulus, root_order)] = False
        if (
            isinstance(val, int)
            and isinstance(modulus, int)
            and isinstance(root_order, int)
            and modulus >= 2
            and root_order >= 1
        ):
            is_rou: bool = is_root_of_unity(
                val=val, modulus=modulus, root_order=root_order
            )
            is_prim: bool = is_rou and all(
                pow(val, i, modulus) != 1 for i in range(1, root_order)
            )
            CACHED_IS_PRIMITIVE_ROOT_OF_UNITY[(val, modulus, root_order)] = is_prim
    return CACHED_IS_PRIMITIVE_ROOT_OF_UNITY[(val, modulus, root_order)]


def find_primitive_root(modulus: int, root_order: int) -> int:
    """
    Find a primitive root of order root_order modulo modulus. Naive loop that first checks 2, then 3, then 4...
    :param modulus: Input modulus
    :type modulus: int
    :param root_order: Order of the root of unity
    :type root_order: int
    :return r: root of unity r such that r**root_order == 1 modulo q and r**i != 1 modulo q for all i in range(1, root_order)
    :rtype: int
    """
    if (modulus, root_order) not in CACHED_FIND_PRIMITIVE_ROOT:
        CACHED_FIND_PRIMITIVE_ROOT[(modulus, root_order)] = None
        if (
            isinstance(modulus, int)
            and isinstance(root_order, int)
            and modulus >= 2
            and root_order >= 1
            and has_primitive_root_of_unity(modulus=modulus, root_order=root_order)
        ):
            r: int = 2
            while r < modulus and not is_primitive_root(
                val=r, modulus=modulus, root_order=root_order
            ):
                r += 1
            if not is_primitive_root(val=r, modulus=modulus, root_order=root_order):
                raise RuntimeError(
                    f"No primitive root found with modulus={modulus}, root_order={root_order}. This error should never occur."
                )
            CACHED_FIND_PRIMITIVE_ROOT[(modulus, root_order)] = r
    return CACHED_FIND_PRIMITIVE_ROOT[(modulus, root_order)]


def cooley_tukey_ntt(
    val: List[int], modulus: int, root_order: int, bit_rev_root_powers: List[int]
) -> List[int]:
    """
    Input val, a list of n := len(val) integers in usual ordering, a modulus that is a prime such that
    (modulus-1) % (2*n) == 0, a root_order == 2*n, and a list of integers, root_powers, which are powers of a primitive
    root of unity with order root_order stored in bit-reversed order.  Output the NTT of val in bit-reversed order.

    In-place computation, iterative implementation of the Cooley-Tukey butterfly as defined in 'Speeding up the Number
    Theoretic Transform for Faster Ideal Lattice-Based Cryptography' by Longa and Naehrig
    (https://eprint.iacr.org/2016/504.pdf).

    :param val: Input list of integers
    :type val: List[int]
    :param modulus:
    :type modulus: int
    :param root_order:
    :type root_order: int
    :param bit_rev_root_powers:
    :type bit_rev_root_powers: List[int]
    :return val: Output list of integers
    :rtype: List[int]
    """
    if not isinstance(val, list):
        raise TypeError(f"val must be a list, but got {type(val)}")
    elif not isinstance(modulus, int):
        raise TypeError(f"modulus must be an int, but got {type(modulus)}")
    elif not isinstance(bit_rev_root_powers, list):
        raise TypeError(
            f"root_powers must be a list, but got {type(bit_rev_root_powers)}"
        )
    elif not all(isinstance(v, int) for v in bit_rev_root_powers):
        raise TypeError(
            f"root_powers must be a list of ints, but got {type(bit_rev_root_powers)}"
        )
    elif not isinstance(root_order, int):
        raise TypeError(f"root_order must be an int, but got {type(root_order)}")
    elif not all(isinstance(v, int) for v in val):
        raise TypeError(f"val must be a list of ints, but got {type(val)}")
    elif not is_odd_prime(val=modulus):
        raise ValueError(f"modulus={modulus} must be an odd prime")
    elif not has_primitive_root_of_unity(modulus=modulus, root_order=root_order):
        raise ValueError(
            f"modulus={modulus} does not have a primitive root of order root_order={root_order}"
        )
    elif not is_pow_two_geq_two(val=len(val)):
        raise ValueError(f"len(val)={len(val)} must be a power of 2 greater than 1")
    elif root_order != 2 * len(val) and root_order != len(val):
        raise ValueError(
            f"root_order={root_order} must be degree or twice the degree, {len(val)}"
        )
    elif root_order == len(val):
        raise NotImplementedError(
            f"root_order={root_order}=degree={len(val)} is not implemented"
        )
    halfmod: int = modulus // 2
    logmod: int = ceil(log2(modulus))
    n: int = len(val)
    t = n
    m = 1
    while m < len(val):
        t //= 2
        for i in range(m):
            j_one = 2 * i * t
            j_two = j_one + t - 1
            s = bit_rev_root_powers[m + i]
            for j in range(j_one, j_two + 1):
                u, v = val[j], val[j + t] * s
                val[j] = cent(
                    val=u + v, modulus=modulus, halfmod=halfmod, logmod=logmod
                )
                val[j + t] = cent(
                    val=u - v, modulus=modulus, halfmod=halfmod, logmod=logmod
                )
        m *= 2
    return val


def gentleman_sande_intt(
    val: List[int], modulus: int, root_order: int, bit_rev_inv_root_powers: List[int]
) -> List[int]:
    """
    Input val, a list of n := len(val) integers in usual ordering, a modulus that is a prime such that
    (modulus-1) % (2*n) == 0, a root_order == 2*n, and a list of integers, inv_root_powers, which are powers of the
    inverse of the primitive root of unity with order root_order used to compute the forward transform, stored in bit-
    reversed order.  Output the INTT of val in standard order.

    In-place computation, iterative implementation of the Gentleman-Sande butterfly as in 'Speeding up the Number
    Theoretic Transform for Faster Ideal Lattice-Based Cryptography' by Longa and Naehrig
    (https://eprint.iacr.org/2016/504.pdf).

    :param val: Input list of integers
    :type val: List[int]
    :param modulus:
    :type modulus: int
    :param root_order:
    :type root_order: int
    :param bit_rev_inv_root_powers:
    :type bit_rev_inv_root_powers: List[int]
    :return val: Output list of integers
    :rtype: List[int]
    """
    if not isinstance(val, list):
        raise TypeError(f"val must be a list, but got {type(val)}")
    elif not isinstance(modulus, int):
        raise TypeError(f"modulus must be an int, but got {type(modulus)}")
    elif not isinstance(bit_rev_inv_root_powers, list):
        raise TypeError(
            f"inv_root_powers must be a list, but got {type(bit_rev_inv_root_powers)}"
        )
    elif not all(isinstance(v, int) for v in bit_rev_inv_root_powers):
        raise TypeError(
            f"inv_root_powers must be a list of ints, but got {type(bit_rev_inv_root_powers)}"
        )
    elif not isinstance(root_order, int):
        raise TypeError(f"root_order must be an int, but got {type(root_order)}")
    elif not all(isinstance(v, int) for v in val):
        raise TypeError(f"val must be a list of ints, but got {type(val)}")
    elif not is_odd_prime(val=modulus):
        raise ValueError(f"modulus={modulus} must be an odd prime")
    elif not has_primitive_root_of_unity(modulus=modulus, root_order=root_order):
        raise ValueError(
            f"modulus={modulus} does not have a primitive root of order root_order={root_order}"
        )
    elif not is_pow_two_geq_two(val=len(val)):
        raise ValueError(f"len(val)={len(val)} must be a power of 2 greater than 1")
    elif root_order != 2 * len(val) and root_order != len(val):
        raise ValueError(
            f"root_order={root_order} must be degree or twice the degree, {len(val)}"
        )
    elif root_order == len(val):
        raise NotImplementedError(
            f"root_order={root_order}=degree={len(val)} is not implemented"
        )
    halfmod: int = modulus // 2
    logmod: int = ceil(log2(modulus))
    n: int = len(val)
    n_inv: int = pow(n, modulus - 2, modulus)
    t = 1
    m = n
    while m > 1:
        j_one = 0
        h = m // 2
        for i in range(h):
            j_two = j_one + t - 1
            s = bit_rev_inv_root_powers[h + i]
            for j in range(j_one, j_two + 1):
                u, v = val[j], val[j + t]
                val[j] = cent(
                    val=u + v, modulus=modulus, halfmod=halfmod, logmod=logmod
                )
                val[j + t] = cent(
                    val=(u - v) * s, modulus=modulus, halfmod=halfmod, logmod=logmod
                )
            j_one += 2 * t
        t *= 2
        m = h
    for j in range(n):
        val[j] = cent(
            val=val[j] * n_inv, modulus=modulus, halfmod=halfmod, logmod=logmod
        )
    return val


def ntt_poly_mult(
    f: List[int], g: List[int], modulus: int, root: int, inv_root: int, root_order: int
) -> List[int]:
    """
    Input two coefficient representations of polynomials, f(X), g(X), and output INTT(NTT(f(X)) * NTT(g(X))). Depends on
    Input should have:
        - modulus is an odd prime
        - root_order is a power of 2
        - root_order >= 2
        - root_order//2 == len(f) == len(g)
        - has_primitive_root_of_unity(modulus, root_order) == True
        - is_primitive_root(root, modulus, root_order) == True
        - (root * root_inv) % modulus == 1
    :param f: Coefficient representation of polynomial f(X)
    :type f: List[int]
    :param g: Coefficient representation of polynomial g(X)
    :type g: List[int]
    :param modulus: Integer modulus
    :type modulus: int
    :param root: Root of unity
    :type root: int | None
    :param inv_root: Inverse of the root of unity
    :type inv_root: int | None
    :param root_order: Order of the root of unity
    :type root_order: int
    :return fg:
    :rtype: List[int]
    """
    if (
        not isinstance(f, list)
        or not isinstance(g, list)
        or not isinstance(modulus, int)
        or not isinstance(root, int)
        or not isinstance(inv_root, int)
        or not isinstance(root_order, int)
    ):
        raise ValueError(
            "Input f and g must be lists of integers, input modulus must be integer, and input root and inv_root must be integer."
        )
    elif not is_odd_prime(val=modulus):
        raise ValueError("Modulus must be an odd prime.")
    elif not is_pow_two_geq_two(val=root_order):
        raise ValueError(
            "Root order must be a power of two greater than or equal to 2."
        )
    elif not len(f) == len(g) == root_order // 2:
        raise ValueError(
            f"f and g must be coefficient representation of degree root_order//2 - 1 polynomial, but had len(f)={len(f)}, len(g)={len(g)}"
        )
    elif not has_primitive_root_of_unity(modulus=modulus, root_order=root_order):
        raise ValueError(
            "Modulus does not have a primitive root of unity of order root_order."
        )
    elif not is_primitive_root(val=root, modulus=modulus, root_order=root_order):
        raise ValueError("Input root must be a primitive root of unity.")
    elif not (root * inv_root) % modulus == 1:
        raise ValueError("Input inv_root must be the inverse of the root of unity.")
    halfmod: int = modulus // 2
    logmod: int = modulus.bit_length()
    n: int = len(f)
    r: Optional[int] = root
    if r is None:
        r: int = find_primitive_root(modulus=modulus, root_order=root_order)
    root_powers: List[int] = [pow(r, i, modulus) for i in range(n)]
    bit_rev_root_powers: List[int] = bit_reverse_copy(val=root_powers)
    inv_r: Optional[int] = inv_root
    if inv_r is None:
        inv_r: int = pow(r, modulus - 2, modulus)
    inv_root_powers: List[int] = [pow(inv_r, i, modulus) for i in range(n)]
    bit_rev_inv_root_powers = bit_reverse_copy(val=inv_root_powers)
    cooley_tukey_ntt(
        val=f,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )
    cooley_tukey_ntt(
        val=g,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )
    fg: List[int] = [
        cent(val=x * y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for x, y in zip(f, g)
    ]
    gentleman_sande_intt(
        val=fg,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )
    gentleman_sande_intt(
        val=f,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )
    gentleman_sande_intt(
        val=g,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )
    return fg


def ntt_poly_mult_half(
    f: List[int], g: List[int], modulus: int, root: int, inv_root: int, root_order: int
) -> List[int]:
    if (
        not isinstance(f, list)
        or not isinstance(g, list)
        or not isinstance(modulus, int)
        or not isinstance(root, int)
        or not isinstance(inv_root, int)
        or not isinstance(root_order, int)
    ):
        raise ValueError(
            "Input f and g must be lists of integers, input modulus must be integer, and input root and inv_root must be integer."
        )
    elif not is_odd_prime(val=modulus):
        raise ValueError("Modulus must be an odd prime.")
    elif not is_pow_two_geq_two(val=root_order):
        raise ValueError(
            "Root order must be a power of two greater than or equal to 2."
        )
    elif not len(f) == len(g) == root_order:
        raise ValueError(
            "Polynomials must have equal length and length equal to root_order."
        )
    elif not has_primitive_root_of_unity(modulus=modulus, root_order=root_order):
        raise ValueError(
            "Modulus does not have a primitive root of unity of order root_order."
        )
    elif not is_primitive_root(val=root, modulus=modulus, root_order=root_order):
        raise ValueError("Input root must be a primitive root of unity.")
    elif not (root * inv_root) % modulus == 1:
        raise ValueError("Input inv_root must be the inverse of the root of unity.")
    halfmod: int = modulus // 2
    logmod: int = ceil(log2(modulus))
    root_powers: List[int] = [pow(root, i, modulus) for i in range(root_order)]
    bit_rev_root_powers: List[int] = bit_reverse_copy(val=root_powers)
    inv_root_powers: List[int] = [pow(inv_root, i, modulus) for i in range(root_order)]
    bit_rev_inv_root_powers: List[int] = bit_reverse_copy(val=inv_root_powers)

    f_evens: List[int] = f[::2]
    f_odds: List[int] = f[1::2]
    g_evens: List[int] = g[::2]
    g_odds: List[int] = g[1::2]

    cooley_tukey_ntt(
        val=f_evens,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )
    cooley_tukey_ntt(
        val=f_odds,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )
    cooley_tukey_ntt(
        val=g_evens,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )
    cooley_tukey_ntt(
        val=g_odds,
        modulus=modulus,
        root_order=root_order,
        bit_rev_root_powers=bit_rev_root_powers,
    )

    f_evens_times_g_evens: List[int] = [
        cent(val=x * y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for x, y in zip(f_evens, g_evens)
    ]
    f_evens_times_g_odds: List[int] = [
        cent(val=x * y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for x, y in zip(f_evens, g_odds)
    ]
    f_odds_times_g_evens: List[int] = [
        cent(val=x * y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for x, y in zip(f_odds, g_evens)
    ]
    f_odds_times_g_odds: List[int] = [
        cent(val=r * x * y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for r, x, y in zip(bit_rev_root_powers, f_odds, g_odds)
    ]

    fg_evens: List[int] = [
        cent(val=x + y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for r, x, y in zip(f_evens_times_g_evens, f_odds_times_g_odds)
    ]
    fg_odds: List[int] = [
        cent(val=x + y, modulus=modulus, halfmod=halfmod, logmod=logmod)
        for x, y in zip(f_evens_times_g_odds, f_odds_times_g_evens)
    ]

    gentleman_sande_intt(
        val=fg_evens,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )
    gentleman_sande_intt(
        val=fg_odds,
        modulus=modulus,
        root_order=root_order,
        bit_rev_inv_root_powers=bit_rev_inv_root_powers,
    )

    fg: List[int] = [elem for pair in zip(fg_evens, fg_odds) for elem in pair]
    return fg

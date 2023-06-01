from copy import deepcopy
from random import seed as random_seed, randrange
from typing import Dict, List, Union, Optional

from algebra.ntt import (
    cent,
    cooley_tukey_ntt,
    gentleman_sande_intt,
    bit_reverse_copy,
)

cached_halfmods: Dict[int, int] = {}
cached_logmods: Dict[int, int] = {}


class PolynomialRepresentation(object):
    modulus: int
    degree: int
    root: int
    inv_root: int
    root_order: int

    def __init__(
        self, modulus: int, degree: int, root: int, inv_root: int, root_order: int
    ):
        if not isinstance(modulus, int):
            raise TypeError("modulus must be an int")
        elif not isinstance(degree, int):
            raise TypeError("degree must be an int")
        elif not isinstance(root, int):
            raise TypeError("root must be an int")
        elif not isinstance(inv_root, int):
            raise TypeError("inv_root must be an int")
        elif not isinstance(root_order, int):
            raise TypeError("root_order must be an int")
        elif (modulus - 1) % root_order != 0:
            raise ValueError("root_order must be a divisor of modulus - 1")
        elif pow(root, root_order, modulus) != 1:
            raise ValueError("root must be a root of unity of order root_order")
        elif any(pow(root, i, modulus) == 1 for i in range(1, root_order)):
            raise ValueError(
                "root must be a primitive root of unity of order root_order"
            )
        elif (root * inv_root) % modulus != 1:
            raise ValueError("root and inv_root must be inverses of each other")
        self.modulus = modulus
        self.degree = degree
        self.root = root
        self.inv_root = inv_root
        self.root_order = root_order

    @property
    def halfmod(self) -> int:
        if self.modulus not in cached_halfmods:
            cached_halfmods[self.modulus] = self.modulus // 2
        return cached_halfmods[self.modulus]

    @property
    def logmod(self) -> int:
        if self.modulus not in cached_logmods:
            cached_logmods[self.modulus] = self.modulus.bit_length() - 1
        return cached_logmods[self.modulus]


class PolynomialCoefficientRepresentation(PolynomialRepresentation):
    coefficients: List[int]

    def __init__(
        self,
        modulus: int,
        degree: int,
        root: int,
        inv_root: int,
        root_order: int,
        coefficients: List[int],
    ):
        super().__init__(
            modulus=modulus,
            degree=degree,
            root=root,
            inv_root=inv_root,
            root_order=root_order,
        )
        if not isinstance(coefficients, list):
            raise TypeError("coefficients must be a list")
        elif not all(isinstance(x, int) for x in coefficients):
            raise TypeError("coefficients must be a list of ints")
        elif len(coefficients) != degree:
            raise ValueError("coefficients must be of length degree")
        self.coefficients = coefficients

    def __str__(self):
        return f"PolynomialCoefficientRepresentation(modulus={self.modulus}, degree={self.degree}, root={self.root}, inv_root={self.inv_root}, root_order={self.root_order}, coefficients={self.coefficients})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, PolynomialCoefficientRepresentation):
            return False
        elif self.modulus != other.modulus:
            return False
        elif self.degree != other.degree:
            return False
        elif self.root != other.root:
            return False
        elif self.root_order != other.root_order:
            return False
        return all(
            (x - y) % self.modulus == 0
            for x, y in zip(self.coefficients, other.coefficients)
        )

    def __add__(self, other):
        if other == 0:
            return self
        elif not isinstance(other, PolynomialCoefficientRepresentation):
            raise NotImplementedError(
                f"Addition for {type(self)} and {type(other)} not implemented"
            )
        elif self.modulus != other.modulus:
            raise NotImplementedError("Cannot add polynomials with different moduli")
        elif self.degree != other.degree:
            raise NotImplementedError("Cannot add polynomials with different degrees")
        elif self.root != other.root:
            raise NotImplementedError(
                "Cannot add polynomials with different roots of unity"
            )
        elif self.root_order != other.root_order:
            raise NotImplementedError(
                "Cannot add polynomials with different root orders"
            )
        return PolynomialCoefficientRepresentation(
            modulus=self.modulus,
            degree=self.degree,
            root=self.root,
            inv_root=self.inv_root,
            root_order=self.root_order,
            coefficients=[
                cent(
                    val=x + y,
                    modulus=self.modulus,
                    halfmod=self.halfmod,
                    logmod=self.logmod,
                )
                for x, y in zip(self.coefficients, other.coefficients)
            ],
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __neg__(self):
        return PolynomialCoefficientRepresentation(
            modulus=self.modulus,
            degree=self.degree,
            root=self.root,
            inv_root=self.inv_root,
            root_order=self.root_order,
            coefficients=[-(x % self.modulus) for x in self.coefficients],
        )

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if other == 0:
            return 0
        elif other == 1:
            return self
        elif not isinstance(other, PolynomialCoefficientRepresentation):
            raise NotImplementedError(
                f"Multiplication for {type(self)} and {type(other)} not implemented"
            )
        elif self.modulus != other.modulus:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with different moduli not implemented"
            )
        elif self.degree != other.degree:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with different degrees not implemented"
            )
        elif self.root != other.root:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with different roots of unity not implemented"
            )
        elif self.root_order != other.root_order:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with difference orders of root of unity not implemented"
            )
        c: List[int] = [0 for _ in range(2 * self.degree)]
        for i, x in enumerate(self.coefficients):
            for j, y in enumerate(other.coefficients):
                c[i + j] += x * y
        c: List[int] = [
            cent(
                val=x - y,
                modulus=self.modulus,
                halfmod=self.halfmod,
                logmod=self.logmod,
            )
            for x, y in zip(c[: self.degree], c[self.degree :])
        ]
        return PolynomialCoefficientRepresentation(
            modulus=self.modulus,
            degree=self.degree,
            root=self.root,
            inv_root=self.inv_root,
            root_order=self.root_order,
            coefficients=c,
        )

    def __rmul__(self, other):
        return self.__mul__(other=other)

    def norm(self, p: Union[int, str]) -> int:
        if p != "infty":
            raise NotImplementedError(f"norm for p={p} not implemented")
        return max(abs(x) for x in self.coefficients)

    def weight(self) -> int:
        return sum(1 if x % self.modulus != 0 else 0 for x in self.coefficients)


class PolynomialNTTRepresentation(PolynomialRepresentation):
    values: List[int]

    def __init__(
        self,
        modulus: int,
        degree: int,
        root: int,
        inv_root: int,
        root_order: int,
        values: List[int],
    ):
        super().__init__(
            modulus=modulus,
            degree=degree,
            root=root,
            inv_root=inv_root,
            root_order=root_order,
        )
        if not isinstance(values, list):
            raise TypeError("values must be a list")
        elif not all(isinstance(x, int) for x in values):
            raise TypeError("values must be a list of ints")
        elif len(values) != degree:
            raise ValueError("values must have length degree")
        self.values = values

    def __str__(self):
        return f"PolynomialNTTRepresentation(modulus={self.modulus}, degree={self.degree}, root={self.root}, inv_root={self.inv_root}, root_order={self.root_order}, values={self.values})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other == 0:
            return all(x % self.modulus == 0 for x in self.values)
        elif not isinstance(other, PolynomialNTTRepresentation):
            return False
        elif self.modulus != other.modulus:
            return False
        elif self.degree != other.degree:
            return False
        elif self.root_order != other.root_order:
            return False
        elif self.root != other.root or self.inv_root != other.inv_root:
            return False
        elif len(self.values) != len(other.values):
            return False
        return all(
            (x - y) % self.modulus == 0 for x, y in zip(self.values, other.values)
        )

    def __add__(self, other):
        if other == 0:
            return self
        elif not isinstance(other, PolynomialNTTRepresentation):
            raise NotImplementedError(
                f"Addition for {type(self)} and {type(other)} not implemented"
            )
        elif self.modulus != other.modulus:
            raise NotImplementedError("Cannot add polynomials with different moduli")
        elif self.degree != other.degree:
            raise NotImplementedError("Cannot add polynomials with different degrees")
        elif self.root != other.root:
            raise NotImplementedError(
                "Cannot add polynomials with different roots of unity"
            )
        elif self.root_order != other.root_order:
            raise NotImplementedError(
                "Cannot add polynomials with different root orders"
            )
        elif len(self.values) != len(other.values):
            raise NotImplementedError("Cannot add polynomials with different lengths")
        return PolynomialNTTRepresentation(
            modulus=self.modulus,
            degree=self.degree,
            root=self.root,
            inv_root=self.inv_root,
            root_order=self.root_order,
            values=[
                cent(
                    val=x + y,
                    modulus=self.modulus,
                    halfmod=self.halfmod,
                    logmod=self.logmod,
                )
                for x, y in zip(self.values, other.values)
            ],
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __neg__(self):
        return PolynomialNTTRepresentation(
            modulus=self.modulus,
            degree=self.degree,
            root=self.root,
            inv_root=self.inv_root,
            root_order=self.root_order,
            values=[-(x % self.modulus) for x in self.values],
        )

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if other == 0:
            return 0
        elif other == 1:
            return self
        elif not isinstance(other, PolynomialNTTRepresentation):
            raise NotImplementedError(
                f"Multiplication for {type(self)} and {type(other)} not implemented"
            )
        elif self.modulus != other.modulus:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with different moduli not implemented"
            )
        elif self.degree != other.degree:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with different degrees not implemented"
            )
        elif self.root != other.root:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with different roots of unity not implemented"
            )
        elif self.root_order != other.root_order:
            raise NotImplementedError(
                f"Multiplication for {type(self)} with difference orders of root of unity not implemented"
            )
        elif len(self.values) != len(other.values):
            raise NotImplementedError(
                f"Multiplication for {type(self)} with different lengths not implemented"
            )
        return PolynomialNTTRepresentation(
            modulus=self.modulus,
            degree=self.degree,
            root=self.root,
            inv_root=self.inv_root,
            root_order=self.root_order,
            values=[
                cent(
                    val=x * y,
                    modulus=self.modulus,
                    halfmod=self.halfmod,
                    logmod=self.logmod,
                )
                for x, y in zip(self.values, other.values)
            ],
        )

    def __rmul__(self, other):
        return self.__mul__(other=other)


def transform(
    x: Union[PolynomialCoefficientRepresentation, PolynomialNTTRepresentation],
) -> Union[PolynomialNTTRepresentation, PolynomialCoefficientRepresentation]:
    if isinstance(x, PolynomialCoefficientRepresentation):
        x_coefs: List[int] = deepcopy(x.coefficients)
        root_powers = [pow(x.root, i, x.modulus) for i in range(x.degree)]
        bit_rev_root_powers = bit_reverse_copy(val=root_powers)
        cooley_tukey_ntt(
            val=x_coefs,
            modulus=x.modulus,
            root_order=x.root_order,
            bit_rev_root_powers=bit_rev_root_powers,
        )
        return PolynomialNTTRepresentation(
            modulus=x.modulus,
            degree=x.degree,
            root=x.root,
            inv_root=x.inv_root,
            root_order=x.root_order,
            values=x_coefs,
        )
    elif isinstance(x, PolynomialNTTRepresentation):
        x_vals: List[int] = deepcopy(x.values)
        root_powers = [pow(x.root, i, x.modulus) for i in range(x.degree)]
        bit_rev_root_powers = bit_reverse_copy(val=root_powers)
        inv_root_powers = [pow(x.inv_root, i, x.modulus) for i in range(x.degree)]
        bit_rev_inv_root_powers = bit_reverse_copy(val=inv_root_powers)
        gentleman_sande_intt(
            val=x_vals,
            modulus=x.modulus,
            root_order=x.root_order,
            bit_rev_inv_root_powers=bit_rev_inv_root_powers,
        )
        return PolynomialCoefficientRepresentation(
            modulus=x.modulus,
            degree=x.degree,
            root=x.root,
            inv_root=x.inv_root,
            root_order=x.root_order,
            coefficients=x_vals,
        )
    else:
        raise NotImplementedError(f"Transform for {type(x)} not implemented")


def sample_polynomial_coefficient_representation(
    modulus: int,
    degree: int,
    root: int,
    inv_root: int,
    root_order: int,
    norm_bound: int,
    weight_bound: int,
    seed: Optional[int],
) -> PolynomialCoefficientRepresentation:
    # Exactly weight non-zero coefficients
    if seed is not None:
        random_seed(seed)
    num_coefs_to_gen: int = max(0, min(degree, weight_bound))
    bound: int = max(0, min(modulus // 2, norm_bound))
    coefficients: List[int] = [
        (1 + randrange(bound)) * (1 - 2 * randrange(2)) for _ in range(num_coefs_to_gen)
    ]
    coefficients += [0 for _ in range(degree - len(coefficients))]
    if num_coefs_to_gen < degree:
        # fisher-yates shuffle
        for i in range(degree - 1, 0, -1):
            j = randrange(i + 1)
            coefficients[i], coefficients[j] = coefficients[j], coefficients[i]
    return PolynomialCoefficientRepresentation(
        modulus=modulus,
        degree=degree,
        root=root,
        inv_root=inv_root,
        root_order=root_order,
        coefficients=coefficients,
    )


def sample_polynomial_ntt_representation(
    modulus: int,
    degree: int,
    root: int,
    inv_root: int,
    root_order: int,
    seed: Optional[int],
) -> PolynomialNTTRepresentation:
    if seed is not None:
        random_seed(seed)
    values: List[int] = [randrange(modulus) - (modulus // 2) for _ in range(degree)]
    return PolynomialNTTRepresentation(
        modulus=modulus,
        degree=degree,
        root=root,
        inv_root=inv_root,
        root_order=root_order,
        values=values,
    )

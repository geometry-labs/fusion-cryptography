from hashlib import shake_256, sha3_256
from math import ceil, log2
from typing import List, Optional, Tuple

from algebra.matrices import GeneralMatrix
from algebra.polynomials import (
    PolynomialCoefficientRepresentation,
    PolynomialNTTRepresentation,
    sample_polynomial_ntt_representation,
    transform,
    sample_polynomial_coefficient_representation,
)

# This implementation is not secure for prod, toy prototype only.

PREFIX_PARAMETERS: dict = {}
PRIME: int = 2147465729
DEGREE_128: int = 2**6
DEGREE_256: int = 2**8
ROOT_ORDER_128: int = 2 * DEGREE_128
ROOT_ORDER_256: int = 2 * DEGREE_256
RANK_128: int = 195
RANK_256: int = 83
CAPACITY_128: int = 1796
CAPACITY_256: int = 2818
CH_WEIGHT_128: int = 27
CH_WEIGHT_256: int = 60
AG_WEIGHT_128: int = 35
AG_WEIGHT_256: int = 60
SK_BD_128: int = 52
SK_BD_256: int = 52
CH_BD_128: int = 3
CH_BD_256: int = 1
AG_BD_128: int = 2
AG_BD_256: int = 1
ROOT_128: int = 23584283
ROOT_256: int = 3337519
SIGN_PRE_HASH_DST_128: bytes = (1).to_bytes(byteorder="little", length=1) + (
    0
).to_bytes(byteorder="little", length=1)
SIGN_PRE_HASH_DST_256: bytes = (3).to_bytes(byteorder="little", length=1) + (
    0
).to_bytes(byteorder="little", length=1)
SIGN_HASH_DST_128: bytes = (1).to_bytes(byteorder="little", length=1) + (1).to_bytes(
    byteorder="little", length=1
)
SIGN_HASH_DST_256: bytes = (3).to_bytes(byteorder="little", length=1) + (1).to_bytes(
    byteorder="little", length=1
)
AGG_XOF_DST_128: bytes = (1).to_bytes(byteorder="little", length=1) + (2).to_bytes(
    byteorder="little", length=1
)
AGG_XOF_DST_256: bytes = (3).to_bytes(byteorder="little", length=1) + (2).to_bytes(
    byteorder="little", length=1
)

VF_BD_INTERMEDIATE_128: int = SK_BD_128 * (
    1 + min(DEGREE_128, CH_WEIGHT_128) * CH_BD_128
)
VF_BD_INTERMEDIATE_256: int = SK_BD_256 * (
    1 + min(DEGREE_256, CH_WEIGHT_256) * CH_BD_256
)
VF_BD_128: int = (
    CAPACITY_128 * min(DEGREE_128, AG_WEIGHT_128) * AG_BD_128 * VF_BD_INTERMEDIATE_128
)
VF_BD_256: int = (
    CAPACITY_256 * min(DEGREE_256, AG_WEIGHT_256) * AG_BD_256 * VF_BD_INTERMEDIATE_256
)

# Check these against the parameterization analysis on iacr
PREFIX_PARAMETERS[128] = {
    "capacity": CAPACITY_128,
    "modulus": PRIME,
    "degree": DEGREE_128,
    "root_order": ROOT_ORDER_128,
    "root": ROOT_128,
    "inv_root": pow(ROOT_128, PRIME - 2, PRIME),
    "num_rows_pub_challenge": 1,
    "num_rows_sk": RANK_128,
    "num_rows_vk": 1,
    "num_cols_pub_challenge": RANK_128,
    "num_cols_sk": 1,
    "num_cols_vk": 1,
    "sign_pre_hash_dst": SIGN_PRE_HASH_DST_128,
    "sign_hash_dst": SIGN_HASH_DST_128,
    "agg_xof_dst": AGG_XOF_DST_128,
    "beta_sk": SK_BD_128,
    "beta_ch": 1,
    "beta_ag": 1,
    "omega_sk": DEGREE_128,
    "omega_ch": CH_WEIGHT_128,
    "omega_ag": AG_WEIGHT_128,
    "beta_vf": VF_BD_128,
    "omega_vf": DEGREE_128,
}

PREFIX_PARAMETERS[256] = {
    "capacity": CAPACITY_256,
    "modulus": PRIME,
    "degree": DEGREE_256,
    "root_order": ROOT_ORDER_256,
    "root": ROOT_256,
    "inv_root": pow(ROOT_256, PRIME - 2, PRIME),
    "num_rows_pub_challenge": 1,
    "num_rows_sk": RANK_256,
    "num_rows_vk": 1,
    "num_cols_pub_challenge": RANK_256,
    "num_cols_sk": 1,
    "num_cols_vk": 1,
    "sign_pre_hash_dst": SIGN_PRE_HASH_DST_256,
    "sign_hash_dst": SIGN_HASH_DST_256,
    "agg_xof_dst": AGG_XOF_DST_256,
    "beta_sk": SK_BD_256,
    "beta_ch": 1,
    "beta_ag": 1,
    "omega_sk": DEGREE_256,
    "omega_ch": CH_WEIGHT_256,
    "omega_ag": AG_WEIGHT_256,
    "beta_vf": VF_BD_256,
    "omega_vf": DEGREE_256,
}

for next_secpar in PREFIX_PARAMETERS:
    tmp: int = ceil(
        (
            ceil(log2(2 * PREFIX_PARAMETERS[next_secpar]["beta_ch"] + 1) / 8)
            + next_secpar / 8
        )
    )
    PREFIX_PARAMETERS[next_secpar]["bytes_for_one_coef_bdd_by_beta_ch"] = tmp
    tmp = ceil(
        (
            ceil(log2(2 * PREFIX_PARAMETERS[next_secpar]["beta_ag"] + 1) / 8)
            + next_secpar / 8
        )
    )
    PREFIX_PARAMETERS[next_secpar]["bytes_for_one_coef_bdd_by_beta_ag"] = tmp
    tmp = PREFIX_PARAMETERS[next_secpar]["degree"] * ceil(
        (ceil(log2(PREFIX_PARAMETERS[next_secpar]["degree"]) / 8) + next_secpar / 8)
    )
    PREFIX_PARAMETERS[next_secpar]["bytes_for_poly_shuffle"] = tmp


def sample_coefficient_matrix(
    seed: Optional[int],
    modulus: int,
    degree: int,
    root_order: int,
    root: int,
    inv_root: int,
    num_rows: int,
    num_cols: int,
    norm_bound: int,
    weight_bound: int,
) -> GeneralMatrix:
    return GeneralMatrix(
        matrix=[
            [
                sample_polynomial_coefficient_representation(
                    modulus=modulus,
                    degree=degree,
                    root_order=root_order,
                    root=root,
                    inv_root=inv_root,
                    norm_bound=norm_bound,
                    weight_bound=weight_bound,
                    seed=seed,
                )
                for j in range(num_cols)
            ]
            for i in range(num_rows)
        ]
    )


def sample_ntt_matrix(
    seed: Optional[int],
    modulus: int,
    degree: int,
    root_order: int,
    root: int,
    inv_root: int,
    num_rows: int,
    num_cols: int,
) -> GeneralMatrix:
    return GeneralMatrix(
        matrix=[
            [
                sample_polynomial_ntt_representation(
                    modulus=modulus,
                    degree=degree,
                    root_order=root_order,
                    root=root,
                    inv_root=inv_root,
                    seed=seed,
                )
                for j in range(num_cols)
            ]
            for i in range(num_rows)
        ]
    )


class Params(object):
    secpar: int
    capacity: int
    modulus: int
    degree: int
    root_order: int
    root: int
    inv_root: int
    num_rows_pub_challenge: int
    num_rows_sk: int
    num_rows_vk: int
    num_cols_pub_challenge: int
    num_cols_sk: int
    num_cols_vk: int
    beta_sk: int
    beta_ch: int
    beta_ag: int
    beta_vf: int
    omega_sk: int
    omega_ch: int
    omega_ag: int
    omega_vf: int
    public_challenge: GeneralMatrix
    sign_pre_hash_dst: str
    sign_hash_dst: str
    agg_xof_dst: str
    bytes_for_one_coef_bdd_by_beta_ch: int
    bytes_for_one_coef_bdd_by_beta_ag: int
    bytes_for_poly_shuffle: int

    def __init__(self, secpar: int, seed: Optional[int]):
        if secpar in PREFIX_PARAMETERS:
            self.secpar = secpar
            self.capacity = PREFIX_PARAMETERS[secpar]["capacity"]
            self.modulus = PREFIX_PARAMETERS[secpar]["modulus"]
            self.degree = PREFIX_PARAMETERS[secpar]["degree"]
            self.root_order = PREFIX_PARAMETERS[secpar]["root_order"]
            self.root = PREFIX_PARAMETERS[secpar]["root"]
            self.inv_root = PREFIX_PARAMETERS[secpar]["inv_root"]
            self.num_rows_pub_challenge = PREFIX_PARAMETERS[secpar][
                "num_rows_pub_challenge"
            ]
            self.num_rows_sk = PREFIX_PARAMETERS[secpar]["num_rows_sk"]
            self.num_rows_vk = PREFIX_PARAMETERS[secpar]["num_rows_vk"]
            self.num_cols_pub_challenge = PREFIX_PARAMETERS[secpar][
                "num_cols_pub_challenge"
            ]
            self.num_cols_sk = PREFIX_PARAMETERS[secpar]["num_cols_sk"]
            self.num_cols_vk = PREFIX_PARAMETERS[secpar]["num_cols_vk"]
            self.sign_pre_hash_dst = PREFIX_PARAMETERS[secpar]["sign_pre_hash_dst"]
            self.sign_hash_dst = PREFIX_PARAMETERS[secpar]["sign_hash_dst"]
            self.agg_xof_dst = PREFIX_PARAMETERS[secpar]["agg_xof_dst"]
            self.bytes_for_one_coef_bdd_by_beta_ch = PREFIX_PARAMETERS[secpar][
                "bytes_for_one_coef_bdd_by_beta_ch"
            ]
            self.bytes_for_one_coef_bdd_by_beta_ag = PREFIX_PARAMETERS[secpar][
                "bytes_for_one_coef_bdd_by_beta_ag"
            ]
            self.bytes_for_poly_shuffle = PREFIX_PARAMETERS[secpar][
                "bytes_for_poly_shuffle"
            ]
            self.beta_sk = PREFIX_PARAMETERS[secpar]["beta_sk"]
            self.beta_ch = PREFIX_PARAMETERS[secpar]["beta_ch"]
            self.beta_ag = PREFIX_PARAMETERS[secpar]["beta_ag"]
            self.beta_vf = PREFIX_PARAMETERS[secpar]["beta_vf"]
            self.omega_sk = PREFIX_PARAMETERS[secpar]["omega_sk"]
            self.omega_ch = PREFIX_PARAMETERS[secpar]["omega_ch"]
            self.omega_ag = PREFIX_PARAMETERS[secpar]["omega_ag"]
            self.omega_vf = PREFIX_PARAMETERS[secpar]["omega_vf"]
            self.public_challenge = sample_ntt_matrix(
                seed=seed,
                modulus=self.modulus,
                degree=self.degree,
                root_order=self.root_order,
                root=self.root,
                inv_root=self.inv_root,
                num_rows=self.num_rows_pub_challenge,
                num_cols=self.num_cols_pub_challenge,
            )

    def __str__(self) -> str:
        return f"Params(secpar={self.secpar}, capacity={self.capacity}, modulus={self.modulus}, degree={self.degree}, root_order={self.root_order}, root={self.root}, inv_root={self.inv_root}, num_rows_pub_challenge={self.num_rows_pub_challenge}, num_rows_sk={self.num_rows_sk}, num_rows_vk={self.num_rows_vk}, num_cols_pub_challenge={self.num_cols_pub_challenge}, num_cols_sk={self.num_cols_sk}, num_cols_vk={self.num_cols_vk}, beta_sk={self.beta_sk}, beta_ch={self.beta_ch}, beta_ag={self.beta_ag}, beta_vf={self.beta_vf}, omega_sk={self.omega_sk}, omega_ch={self.omega_ch}, omega_ag={self.omega_ag}, omega_vf={self.omega_vf}, public_challenge={str(self.public_challenge)}, sign_pre_hash_dst={self.sign_pre_hash_dst}, sign_hash_dst={self.sign_hash_dst}, agg_xof_dst={self.agg_xof_dst}, bytes_for_one_coef_bdd_by_beta_ch={self.bytes_for_one_coef_bdd_by_beta_ch}, bytes_for_one_coef_bdd_by_beta_ag={self.bytes_for_one_coef_bdd_by_beta_ag}, bytes_for_poly_shuffle={self.bytes_for_poly_shuffle})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def fusion_setup(secpar: int, seed: Optional[int]) -> Params:
    return Params(secpar=secpar, seed=seed)


class OneTimeSigningKey(object):
    seed: Optional[int]
    left_sk_hat: GeneralMatrix
    right_sk_hat: GeneralMatrix

    def __init__(
        self,
        seed: Optional[int],
        left_sk_hat: GeneralMatrix,
        right_sk_hat: GeneralMatrix,
    ):
        self.seed = seed
        self.left_sk_hat = left_sk_hat
        self.right_sk_hat = right_sk_hat

    def __str__(self):
        return f"OneTimeSigningKey(seed={self.seed}, left_sk_hat={str(self.left_sk_hat)}, right_sk_hat={str(self.right_sk_hat)})"

    def __repr__(self):
        return self.__str__()


class OneTimeVerificationKey(object):
    left_vk_hat: GeneralMatrix
    right_vk_hat: GeneralMatrix

    def __init__(self, left_vk_hat: GeneralMatrix, right_vk_hat: GeneralMatrix):
        self.left_vk_hat = left_vk_hat
        self.right_vk_hat = right_vk_hat

    def __str__(self):
        return f"OneTimeVerificationKey(left_vk_hat={self.left_vk_hat}, right_vk_hat={self.right_vk_hat})"

    def __repr__(self):
        return self.__str__()


OneTimeKeyTuple = Tuple[OneTimeSigningKey, OneTimeVerificationKey]


def keygen(params: Params, seed: Optional[int]) -> OneTimeKeyTuple:
    left_key_coefs: GeneralMatrix = sample_coefficient_matrix(
        seed=seed,
        modulus=params.modulus,
        degree=params.degree,
        root_order=params.root_order,
        root=params.root,
        inv_root=params.inv_root,
        num_rows=params.num_rows_sk,
        num_cols=params.num_cols_sk,
        norm_bound=params.beta_sk,
        weight_bound=params.omega_sk,
    )
    right_key_coefs: GeneralMatrix = sample_coefficient_matrix(
        seed=seed + 1,
        modulus=params.modulus,
        degree=params.degree,
        root_order=params.root_order,
        root=params.root,
        inv_root=params.inv_root,
        num_rows=params.num_rows_sk,
        num_cols=params.num_cols_sk,
        norm_bound=params.beta_sk,
        weight_bound=params.omega_sk,
    )
    left_sk_hat: GeneralMatrix = GeneralMatrix(
        matrix=[[transform(y) for y in z] for z in left_key_coefs.matrix]
    )
    right_sk_hat: GeneralMatrix = GeneralMatrix(
        matrix=[[transform(y) for y in z] for z in right_key_coefs.matrix]
    )
    left_vk_hat: GeneralMatrix = params.public_challenge * left_sk_hat
    right_vk_hat: GeneralMatrix = params.public_challenge * right_sk_hat
    return OneTimeSigningKey(
        seed=seed, left_sk_hat=left_sk_hat, right_sk_hat=right_sk_hat
    ), OneTimeVerificationKey(left_vk_hat=left_vk_hat, right_vk_hat=right_vk_hat)


class SignatureChallenge(object):
    c_hat: PolynomialNTTRepresentation

    def __init__(self, c_hat: PolynomialNTTRepresentation):
        self.c_hat = c_hat

    def __str__(self):
        return f"SignatureChallenge(c_hat={str(self.c_hat)})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.c_hat == other.c_hat


class Signature(object):
    signature_hat: GeneralMatrix

    def __init__(self, signature_hat: GeneralMatrix):
        self.signature_hat = signature_hat

    def __str__(self):
        return f"Signature(signature_hat={str(self.signature_hat)})"

    def __repr__(self):
        return self.__str__()


def hash_message_to_int(params: Params, message: str) -> int:
    # hash the message m with keccak/SHA3-256
    salted_message = (params.sign_pre_hash_dst.decode("utf-8") + "," + message).encode()
    pre_hashed_message = sha3_256(salted_message).digest()
    return int.from_bytes(pre_hashed_message, byteorder="little")


def hash_vk_and_int_to_bytes(
    params: Params, key: OneTimeVerificationKey, i: int, n: int
) -> bytes:
    # assuming `key` can be represented as string
    x: bytes = (
        params.sign_hash_dst.decode("utf-8") + "," + str(key) + "," + str(i)
    ).encode("utf-8")
    return shake_256(x).digest(n)


def decode_bytes_to_polynomial_coefficients(
    b: bytes,
    log2_bias: int,
    modulus: int,
    degree: int,
    norm_bound: int,
    weight_bound: int,
) -> List[int]:
    # some convenient data
    num_coefs: int = max(1, min(degree, weight_bound))
    bound: int = max(1, min(modulus // 2, norm_bound))
    bytes_per_coefficient: int = ceil((log2(bound) + 1 + log2_bias) / 8)
    bytes_per_index: int = ceil((log2(degree) + log2_bias) / 8)
    bytes_for_signums: int = ceil(weight_bound / 8)
    total_bytes: int = (
        bytes_for_signums + (bytes_per_coefficient + bytes_per_index) * weight_bound
    )
    if len(b) < total_bytes:
        raise ValueError(
            f"Too few bytes to decode polynomial. Expected {total_bytes} but got {len(b)}"
        )

    # Generate the signums of each coefficient
    signum_bytes: bytes
    signum_bytes, remaining_bytes = b[:bytes_for_signums], b[bytes_for_signums:]
    signums_as_int: int = int.from_bytes(signum_bytes, byteorder="big")
    signums_as_bits: str = bin(signums_as_int)[2:].zfill(8 * bytes_for_signums)[::-1]
    signums = [
        2 * int(next_signum) - 1
        for i, next_signum in enumerate(signums_as_bits)
        if i < weight_bound
    ]

    # Generate the coefficients
    coefficients: List[int] = []
    next_bytes: bytes
    for i in range(weight_bound):
        next_bytes, remaining_bytes = (
            remaining_bytes[:bytes_per_coefficient],
            remaining_bytes[bytes_per_coefficient:],
        )
        coefficients += [
            ((int.from_bytes(next_bytes, byteorder="big") % bound) + 1) * signums[i]
        ]
    coefficients += [
        0 for _ in range(degree - len(coefficients))
    ]  # Pad to correct length if necessary

    # Fisher-Yates shuffle if necessary
    # This part can be optimized a lot
    if num_coefs < degree:
        for i in range(degree - 1, weight_bound, -1):
            # convert next_few_bytes to an integer modulo i+1
            next_bytes, remaining_bytes = (
                remaining_bytes[:bytes_per_index],
                remaining_bytes[bytes_per_index:],
            )
            j: int = int.from_bytes(next_bytes, byteorder="big") % (i + 1)
            coefficients[i], coefficients[j] = coefficients[j], coefficients[i]
    return coefficients


def parse_challenge(params: Params, b: bytes) -> PolynomialNTTRepresentation:
    if (
        len(b)
        < params.omega_ch * params.bytes_for_one_coef_bdd_by_beta_ch
        + params.bytes_for_poly_shuffle
    ):
        raise ValueError("hashed_vk_and_pre_hashed_message is too short")
    c_coefs: List[int] = decode_bytes_to_polynomial_coefficients(
        b=b,
        log2_bias=params.secpar,
        modulus=params.modulus,
        degree=params.degree,
        norm_bound=params.beta_ch,
        weight_bound=params.omega_ch,
    )
    c: PolynomialCoefficientRepresentation = PolynomialCoefficientRepresentation(
        modulus=params.modulus,
        degree=params.degree,
        root=params.root,
        inv_root=params.inv_root,
        root_order=params.root_order,
        coefficients=c_coefs,
    )
    c_hat: PolynomialNTTRepresentation = transform(c)
    return c_hat


def hash_ch(
    params: Params, key: OneTimeVerificationKey, message: str
) -> SignatureChallenge:
    pre_hashed_message: int = hash_message_to_int(params=params, message=message)
    num_coefs: int = max(0, min(params.degree, params.omega_ch))
    bound: int = max(0, min(params.modulus // 2, params.beta_ch))
    bytes_per_coefficient: int = ceil((log2(bound) + 1 + params.secpar) / 8)
    bytes_per_index: int = ceil((log2(params.degree) + params.secpar) / 8)
    bytes_for_signums: int = ceil(params.omega_ch / 8)
    n: int = (
        bytes_for_signums
        + bytes_per_coefficient * num_coefs
        + params.degree * bytes_per_index
    )
    chall_as_bytes: bytes = hash_vk_and_int_to_bytes(
        params=params, key=key, i=pre_hashed_message, n=n
    )
    parsed_chall: PolynomialNTTRepresentation = parse_challenge(
        params=params, b=chall_as_bytes
    )
    return SignatureChallenge(c_hat=parsed_chall)


def sign(params: Params, key: OneTimeKeyTuple, message: str) -> Signature:
    sk: OneTimeSigningKey  # type
    vk: OneTimeVerificationKey  # type
    sk, vk = key  # unpack keys
    pre_hashed_message: int = hash_message_to_int(
        params=params, message=message
    )  # pre-hash the message before hashing with vk
    num_coefs: int = max(0, min(params.degree, params.omega_ch))
    bound: int = max(0, min(params.modulus // 2, params.beta_ch))
    bytes_per_coefficient: int = ceil((log2(bound) + 1 + params.secpar) / 8)
    bytes_per_index: int = ceil((log2(params.degree) + params.secpar) / 8)
    bytes_for_signums: int = ceil(params.omega_ch / 8)
    n: int = (
        bytes_for_signums
        + bytes_per_coefficient * num_coefs
        + params.degree * bytes_per_index
    )
    hashed_vk_and_pre_hashed_message: bytes = hash_vk_and_int_to_bytes(
        params=params, key=vk, i=pre_hashed_message, n=n
    )
    c_hat: PolynomialNTTRepresentation = parse_challenge(
        params=params, b=hashed_vk_and_pre_hashed_message
    )
    return Signature(signature_hat=sk.left_sk_hat * c_hat + sk.right_sk_hat)


class AggregationCoefficient(object):
    alpha_hat: PolynomialNTTRepresentation

    def __init__(self, alpha_hat: PolynomialNTTRepresentation):
        self.alpha_hat = alpha_hat

    def __str__(self):
        return f"AggregationCoefficient(alpha_hat={self.alpha_hat})"

    def __repr__(self):
        return self.__str__()


def hash_vks_and_ints_and_challs_to_bytes(
    params: Params,
    keys: List[OneTimeVerificationKey],
    prehashed_messages: List[int],
    challenges: List[SignatureChallenge],
) -> bytes:
    bound: int = max(0, min(params.modulus // 2, params.beta_ag))
    bytes_per_coefficient: int = ceil((log2(bound) + 1 + params.secpar) / 8)
    bytes_per_index: int = ceil((log2(params.degree) + params.secpar) / 8)
    bytes_for_signums: int = ceil(params.omega_ag / 8)
    n: int = len(keys) * (
        bytes_for_signums + (bytes_per_coefficient + bytes_per_index) * params.omega_ag
    )
    salted_vk_and_pre_hashed_message_as_bytes: bytes = str.encode(
        params.agg_xof_dst.decode("utf-8")
        + ","
        + str(list(zip(keys, prehashed_messages, challenges)))
    )
    return shake_256(salted_vk_and_pre_hashed_message_as_bytes).digest(n)


def decode_bytes_to_agg_coefs(params: Params, b: bytes) -> List[AggregationCoefficient]:
    bound: int = max(0, min(params.modulus // 2, params.beta_ag))
    bytes_per_coefficient: int = ceil((log2(bound) + 1 + params.secpar) / 8)
    bytes_per_index: int = ceil((log2(params.degree) + params.secpar) / 8)
    bytes_for_signums: int = ceil(params.omega_ag / 8)
    n: int = (
        bytes_for_signums + (bytes_per_coefficient + bytes_per_index) * params.omega_ag
    )
    num_agg_coefs: int = len(b) // n
    alpha_hats: List[AggregationCoefficient] = []
    for i in range(num_agg_coefs):
        next_byte_section: bytes = b[i * n : (i + 1) * n]
        next_alpha_coefs: List[int] = decode_bytes_to_polynomial_coefficients(
            b=next_byte_section,
            log2_bias=params.secpar,
            modulus=params.modulus,
            degree=params.degree,
            norm_bound=params.beta_ag,
            weight_bound=params.omega_ag,
        )
        next_alpha: PolynomialCoefficientRepresentation = (
            PolynomialCoefficientRepresentation(
                modulus=params.modulus,
                degree=params.degree,
                root=params.root,
                inv_root=params.inv_root,
                root_order=params.root_order,
                coefficients=next_alpha_coefs,
            )
        )
        next_alpha_hat: PolynomialNTTRepresentation = transform(next_alpha)
        next_aggregation_coefficient: AggregationCoefficient = AggregationCoefficient(
            alpha_hat=next_alpha_hat
        )
        alpha_hats += [next_aggregation_coefficient]
    return alpha_hats


def hash_ag(
    params: Params, keys: List[OneTimeVerificationKey], messages: List[str]
) -> List[AggregationCoefficient]:
    pre_hashed_messages: List[int] = [
        hash_message_to_int(params=params, message=next_m)
        for next_vk, next_m in zip(keys, messages)
    ]
    challs: List[SignatureChallenge] = [
        hash_ch(params=params, key=next_vk, message=next_m)
        for next_vk, next_m in zip(keys, messages)
    ]
    b: bytes = hash_vks_and_ints_and_challs_to_bytes(
        params=params,
        keys=keys,
        prehashed_messages=pre_hashed_messages,
        challenges=challs,
    )
    alpha_hats: List[AggregationCoefficient] = decode_bytes_to_agg_coefs(
        params=params, b=b
    )
    return alpha_hats


def aggregate(
    params: Params,
    keys: List[OneTimeVerificationKey],
    messages: List[str],
    signatures: List[Signature],
) -> Signature:
    sorted_input: List[Tuple[OneTimeVerificationKey, str, Signature]] = sorted(
        list(zip(keys, messages, signatures)), key=lambda x: str(x[0])
    )
    sorted_signatures: List[Signature] = [x[2] for x in sorted_input]
    aggregation_coefficients: List[AggregationCoefficient] = hash_ag(
        params=params,
        keys=[x[0] for x in sorted_input],
        messages=[x[1] for x in sorted_input],
    )
    aggregate_signature_hat_values: GeneralMatrix = (
        sorted_signatures[0].signature_hat * aggregation_coefficients[0].alpha_hat
    )
    for next_alpha, next_sig in zip(
        aggregation_coefficients[1:], sorted_signatures[1:]
    ):
        aggregate_signature_hat_values += next_sig.signature_hat * next_alpha.alpha_hat
    return Signature(signature_hat=aggregate_signature_hat_values)


def verify(
    params: Params,
    keys: List[OneTimeVerificationKey],
    messages: List[str],
    aggregate_signature: Signature,
) -> Tuple[bool, str]:
    if len(keys) > params.capacity:
        return False, f"Too many keys."
    elif len(keys) != len(messages):
        return False, f"Number of keys and messages must be equal."
    coef_rep_agg_sig: GeneralMatrix = GeneralMatrix(
        matrix=[[transform(z) for z in y] for y in aggregate_signature.signature_hat]
    )
    sorted_input = sorted(zip(keys, messages), key=lambda x: str(x[0]))
    sorted_vks: List[OneTimeVerificationKey] = [x[0] for x in sorted_input]
    sorted_challs: List[SignatureChallenge] = [
        hash_ch(params=params, key=next_vk, message=next_m)
        for next_vk, next_m in sorted_input
    ]
    aggregation_coefficients: List[AggregationCoefficient] = hash_ag(
        params=params,
        keys=[x[0] for x in sorted_input],
        messages=[x[1] for x in sorted_input],
    )
    tmp: GeneralMatrix = sorted_vks[0].left_vk_hat * sorted_challs[0].c_hat
    tmp += sorted_vks[0].right_vk_hat
    target: GeneralMatrix = (
        sorted_vks[0].left_vk_hat * sorted_challs[0].c_hat + sorted_vks[0].right_vk_hat
    ) * aggregation_coefficients[0].alpha_hat
    for next_alpha, next_vk, next_chall in zip(
        aggregation_coefficients[1:], sorted_vks[1:], sorted_challs[1:]
    ):
        target += (
            next_vk.left_vk_hat * next_chall.c_hat + next_vk.right_vk_hat
        ) * next_alpha.alpha_hat
    observed: GeneralMatrix = (
        params.public_challenge * aggregate_signature.signature_hat
    )
    for a, b in zip(target.matrix, observed.matrix):
        for c, d in zip(a, b):
            if c != d:
                return False, f"Target doesn't match image of aggregate signature."
    if any(
        z.norm(p="infty") > params.beta_vf for y in coef_rep_agg_sig.matrix for z in y
    ):
        return False, f"Norm of aggregate signature too large."
    elif any(z.weight() > params.omega_vf for y in coef_rep_agg_sig.matrix for z in y):
        return False, f"Weight of aggregate signature too large."
    return True, ""

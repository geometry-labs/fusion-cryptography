import pytest
import os
from copy import deepcopy
from math import ceil, log2
from random import randrange
from typing import List

from algebra.ntt import find_primitive_root
from algebra.polynomials import (
    PolynomialCoefficientRepresentation as Poly,
    PolynomialNTTRepresentation as PolyNTT,
    transform,
)
from fusion.fusion import (
    PREFIX_PARAMETERS,
    sample_coefficient_matrix,
    sample_ntt_matrix,
    fusion_setup,
    keygen,
    sign,
    aggregate,
    verify,
    Params,
    OneTimeKeyTuple,
    OneTimeVerificationKey,
    Signature,
    OneTimeSigningKey,
    GeneralMatrix,
    SignatureChallenge,
    hash_message_to_int,
    hash_vk_and_int_to_bytes,
    parse_challenge,
    hash_ch,
    AggregationCoefficient,
    hash_ag,
    decode_bytes_to_polynomial_coefficients,
)

TEST_SEED: int = 8675309
TEST_SAMPLE_SIZE: int = 2**5


def test_sample_coefficient_matrix():
    # Setup
    params: Params = fusion_setup(secpar=128, seed=TEST_SEED)

    x: GeneralMatrix = sample_coefficient_matrix(
        seed=TEST_SEED,
        modulus=params.modulus,
        degree=params.degree,
        root_order=params.root_order,
        root=params.root,
        inv_root=params.inv_root,
        num_rows=1,
        num_cols=1,
        norm_bound=1,
        weight_bound=1,
    )
    assert len(x.matrix) == 1
    assert all([len(row) == 1 for row in x.matrix])
    assert all(hasattr(z, "norm") for y in x.matrix for z in y)
    assert all(hasattr(z, "weight") for y in x.matrix for z in y)
    assert 0 <= x.norm(p="infty") <= 1
    assert 0 <= x.weight() <= 1

    x: GeneralMatrix = sample_coefficient_matrix(
        seed=TEST_SEED,
        modulus=params.modulus,
        degree=params.degree,
        root_order=params.root_order,
        root=params.root,
        inv_root=params.inv_root,
        num_rows=2,
        num_cols=3,
        norm_bound=17,
        weight_bound=16,
    )
    assert len(x.matrix) == 2
    assert all([len(row) == 3 for row in x.matrix])
    assert all(hasattr(z, "norm") for y in x.matrix for z in y)
    assert all(hasattr(z, "weight") for y in x.matrix for z in y)
    assert 0 <= x.norm(p="infty") <= 17
    assert 0 <= x.weight() <= 16


def test_sample_ntt_matrix():
    # Setup
    params: Params = fusion_setup(secpar=128, seed=TEST_SEED)

    x: GeneralMatrix = sample_ntt_matrix(
        seed=TEST_SEED,
        modulus=params.modulus,
        degree=params.degree,
        root_order=params.root_order,
        root=params.root,
        inv_root=params.inv_root,
        num_rows=1,
        num_cols=1,
    )
    assert len(x.matrix) == 1
    assert all([len(row) == 1 for row in x.matrix])

    x: GeneralMatrix = sample_ntt_matrix(
        seed=TEST_SEED,
        modulus=params.modulus,
        degree=params.degree,
        root_order=params.root_order,
        root=params.root,
        inv_root=params.inv_root,
        num_rows=2,
        num_cols=3,
    )
    assert len(x.matrix) == 2
    assert all([len(row) == 3 for row in x.matrix])


def test_params_and_fusion_setup():
    for next_secpar in [128, 256]:
        expected_params: Params = Params(secpar=next_secpar, seed=TEST_SEED)
        expected_str: str = expected_params.__str__()
        expected_repr: str = expected_params.__repr__()
        observed_params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)
        observed_str: str = observed_params.__str__()
        observed_repr: str = observed_params.__repr__()

        # types
        assert isinstance(expected_params, Params)
        assert isinstance(expected_str, str)
        assert isinstance(expected_repr, str)
        assert isinstance(observed_params, Params)
        assert isinstance(observed_str, str)
        assert isinstance(observed_repr, str)

        # parameters
        assert expected_params.secpar == next_secpar == observed_params.secpar
        assert (
            expected_params.capacity
            == PREFIX_PARAMETERS[next_secpar]["capacity"]
            == observed_params.capacity
        )
        assert (
            expected_params.modulus
            == PREFIX_PARAMETERS[next_secpar]["modulus"]
            == observed_params.modulus
        )
        assert (
            expected_params.degree
            == PREFIX_PARAMETERS[next_secpar]["degree"]
            == observed_params.degree
        )
        assert (
            expected_params.root_order
            == PREFIX_PARAMETERS[next_secpar]["root_order"]
            == observed_params.root_order
        )
        assert (
            expected_params.root
            == PREFIX_PARAMETERS[next_secpar]["root"]
            == observed_params.root
        )
        assert (
            expected_params.inv_root
            == PREFIX_PARAMETERS[next_secpar]["inv_root"]
            == observed_params.inv_root
        )
        assert (
            expected_params.num_rows_pub_challenge
            == PREFIX_PARAMETERS[next_secpar]["num_rows_pub_challenge"]
            == observed_params.num_rows_pub_challenge
        )
        assert (
            expected_params.num_rows_sk
            == PREFIX_PARAMETERS[next_secpar]["num_rows_sk"]
            == observed_params.num_rows_sk
        )
        assert (
            expected_params.num_rows_vk
            == PREFIX_PARAMETERS[next_secpar]["num_rows_vk"]
            == observed_params.num_rows_vk
        )
        assert (
            expected_params.num_cols_pub_challenge
            == PREFIX_PARAMETERS[next_secpar]["num_cols_pub_challenge"]
            == observed_params.num_cols_pub_challenge
        )
        assert (
            expected_params.num_cols_sk
            == PREFIX_PARAMETERS[next_secpar]["num_cols_sk"]
            == observed_params.num_cols_sk
        )
        assert (
            expected_params.num_cols_vk
            == PREFIX_PARAMETERS[next_secpar]["num_cols_vk"]
            == observed_params.num_cols_vk
        )
        assert (
            expected_params.sign_pre_hash_dst
            == PREFIX_PARAMETERS[next_secpar]["sign_pre_hash_dst"]
            == observed_params.sign_pre_hash_dst
        )
        assert (
            expected_params.sign_hash_dst
            == PREFIX_PARAMETERS[next_secpar]["sign_hash_dst"]
            == observed_params.sign_hash_dst
        )
        assert (
            expected_params.agg_xof_dst
            == PREFIX_PARAMETERS[next_secpar]["agg_xof_dst"]
            == observed_params.agg_xof_dst
        )
        assert (
            expected_params.bytes_for_one_coef_bdd_by_beta_ch
            == PREFIX_PARAMETERS[next_secpar]["bytes_for_one_coef_bdd_by_beta_ch"]
            == observed_params.bytes_for_one_coef_bdd_by_beta_ch
        )
        assert (
            expected_params.bytes_for_one_coef_bdd_by_beta_ag
            == PREFIX_PARAMETERS[next_secpar]["bytes_for_one_coef_bdd_by_beta_ag"]
            == observed_params.bytes_for_one_coef_bdd_by_beta_ag
        )
        assert (
            expected_params.bytes_for_poly_shuffle
            == PREFIX_PARAMETERS[next_secpar]["bytes_for_poly_shuffle"]
            == observed_params.bytes_for_poly_shuffle
        )
        assert (
            expected_params.beta_sk
            == PREFIX_PARAMETERS[next_secpar]["beta_sk"]
            == observed_params.beta_sk
        )
        assert (
            expected_params.beta_ch
            == PREFIX_PARAMETERS[next_secpar]["beta_ch"]
            == observed_params.beta_ch
        )
        assert (
            expected_params.beta_ag
            == PREFIX_PARAMETERS[next_secpar]["beta_ag"]
            == observed_params.beta_ag
        )
        assert (
            expected_params.beta_vf
            == PREFIX_PARAMETERS[next_secpar]["beta_vf"]
            == observed_params.beta_vf
        )
        assert (
            expected_params.omega_sk
            == PREFIX_PARAMETERS[next_secpar]["omega_sk"]
            == observed_params.omega_sk
        )
        assert (
            expected_params.omega_ch
            == PREFIX_PARAMETERS[next_secpar]["omega_ch"]
            == observed_params.omega_ch
        )
        assert (
            expected_params.omega_ag
            == PREFIX_PARAMETERS[next_secpar]["omega_ag"]
            == observed_params.omega_ag
        )
        assert (
            expected_params.omega_vf
            == PREFIX_PARAMETERS[next_secpar]["omega_vf"]
            == observed_params.omega_vf
        )

        # __eq__
        assert expected_params == observed_params


def test_key_classes():
    for next_secpar in [128, 256]:
        # Test the OneTimeSigningKey class
        otsk: OneTimeSigningKey = OneTimeSigningKey(
            seed=None, left_sk_hat="Hello world", right_sk_hat="Goodbye world"
        )
        assert otsk.seed is None
        assert otsk.left_sk_hat == "Hello world"
        assert otsk.right_sk_hat == "Goodbye world"
        assert (
            otsk.__str__()
            == f"OneTimeSigningKey(seed={None}, left_sk_hat={'Hello world'}, right_sk_hat={'Goodbye world'})"
        )
        assert otsk.__str__() == otsk.__repr__()

        # Test the OneTimeVerification Class
        otvk: OneTimeVerificationKey = OneTimeVerificationKey(
            left_vk_hat="Hello world", right_vk_hat="Goodbye world"
        )
        assert otvk.left_vk_hat == "Hello world"
        assert otvk.right_vk_hat == "Goodbye world"
        assert (
            otvk.__str__()
            == f"OneTimeVerificationKey(left_vk_hat={'Hello world'}, right_vk_hat={'Goodbye world'})"
        )
        assert otvk.__str__() == otvk.__repr__()

        # Test the OneTimeKeyTuple type
        x: OneTimeKeyTuple = (otsk, otvk)
        assert isinstance(x, tuple)
        assert len(x) == 2
        assert isinstance(x[0], OneTimeSigningKey)
        assert isinstance(x[1], OneTimeVerificationKey)
        assert x[0] == otsk
        assert x[1] == otvk


def test_keygen():
    for next_secpar in [128, 256]:
        # Setup
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)

        # Keygen
        otk: OneTimeKeyTuple = keygen(params=params, seed=TEST_SEED + 1)
        assert isinstance(otk, tuple)
        assert len(otk) == 2
        assert isinstance(otk[0], OneTimeSigningKey)
        assert isinstance(otk[1], OneTimeVerificationKey)

        # Unpack the key
        otsk: OneTimeSigningKey  # private
        otvk: OneTimeVerificationKey  # public
        otsk, otvk = otk  # unpack

        assert hasattr(otsk, "seed")
        assert hasattr(otsk, "left_sk_hat")
        assert hasattr(otsk, "right_sk_hat")
        assert isinstance(otsk.seed, int)
        assert isinstance(otsk.left_sk_hat, GeneralMatrix)
        assert isinstance(otsk.right_sk_hat, GeneralMatrix)

        inv_left_sk_hat: GeneralMatrix = GeneralMatrix(
            matrix=[[transform(f) for f in row] for row in otsk.left_sk_hat.matrix]
        )
        inv_right_sk_hat: GeneralMatrix = GeneralMatrix(
            matrix=[[transform(f) for f in row] for row in otsk.right_sk_hat.matrix]
        )
        assert inv_left_sk_hat.norm(p="infty") <= params.beta_sk
        assert inv_right_sk_hat.norm(p="infty") <= params.beta_sk
        assert inv_left_sk_hat.weight() <= params.omega_sk
        assert inv_right_sk_hat.weight() <= params.omega_sk

        assert hasattr(otvk, "left_vk_hat")
        assert hasattr(otvk, "right_vk_hat")
        assert isinstance(otvk.left_vk_hat, GeneralMatrix)
        assert isinstance(otvk.right_vk_hat, GeneralMatrix)

        assert params.public_challenge * otsk.left_sk_hat == otvk.left_vk_hat
        assert params.public_challenge * otsk.right_sk_hat == otvk.right_vk_hat


def test_signature_challenge_class():
    x: SignatureChallenge = SignatureChallenge(c_hat="Hello world")
    assert isinstance(x, SignatureChallenge)
    assert x.c_hat == "Hello world"
    assert x.__str__() == f"SignatureChallenge(c_hat={'Hello world'})"
    assert x.__str__() == x.__repr__()


def test_signature_class():
    x: Signature = Signature(signature_hat="Hello world")
    assert isinstance(x, Signature)
    assert x.signature_hat == "Hello world"
    assert x.__str__() == f"Signature(signature_hat={'Hello world'})"
    assert x.__str__() == x.__repr__()


def test_hash_message_to_int(mocker):
    for next_secpar in [128, 256]:
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)

        message = "my_message"
        expected_result = 1234567890

        # Create a mock object for the hash object returned by sha3_256
        mock_hash_obj = mocker.Mock()
        mock_hash_obj.digest.return_value = (expected_result).to_bytes(
            32, byteorder="little"
        )

        # Set up the return value for the sha3_256 mock
        mock_sha3_256 = mocker.patch(
            "fusion.fusion.sha3_256", return_value=mock_hash_obj
        )

        # Call the hash_message_to_int function
        result = hash_message_to_int(params, message)

        # Assert that the result matches the expected value
        assert result == expected_result

        # Check that the sha3_256 function was called with the correct arguments
        salted_message = params.sign_pre_hash_dst.decode("utf-8") + "," + message
        mock_sha3_256.assert_called_once_with(salted_message.encode())


def test_hash_vk_and_int_to_bytes(mocker):
    for next_secpar in [128, 256]:
        # Create sample Params and OneTimeVerificationKey objects
        params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)
        keys = keygen(params, seed=TEST_SEED + 1)
        otsk, otvk = keys

        pre_hashed_message = 1234567890

        # Expected result, assuming the shake_256 function would return this value
        expected_result = b"expected_shake_256_result"

        # Create a mock object for the hash object returned by shake_256
        mock_hash_obj = mocker.Mock()
        mock_hash_obj.digest.return_value = expected_result

        # Mock the shake_256 function to return the mock hash object
        mock_shake_256 = mocker.patch(
            "fusion.fusion.shake_256", return_value=mock_hash_obj
        )

        # Call the hash_vk_and_int_to_bytes function
        # we are mocking a result, so n can be anything.
        result = hash_vk_and_int_to_bytes(
            params=params, key=otvk, i=pre_hashed_message, n=1
        )

        # Assert that the result matches the expected value
        assert result == expected_result

        # Check that the shake_256 function was called with the correct arguments
        x = (
            params.sign_hash_dst.decode("utf-8")
            + ","
            + str(otvk)
            + ","
            + str(pre_hashed_message)
        ).encode()
        mock_shake_256.assert_called_once_with(x)


def test_decode_bytes_to_polynomial_coefficients():
    for next_secpar in [128, 256]:
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)
        for next_omega in range(1, 12):
            for next_beta in range(1, 12):
                num_coefs: int = max(0, min(params.degree, next_omega))
                bound: int = max(0, min(params.modulus // 2, next_beta))
                bytes_per_coefficient: int = ceil((log2(bound) + 1 + next_secpar) / 8)
                bytes_per_index: int = ceil((log2(params.degree) + next_secpar) / 8)
                bytes_for_signums: int = ceil(params.omega_ch / 8)
                n: int = (
                    bytes_for_signums
                    + bytes_per_coefficient * num_coefs
                    + params.degree * bytes_per_index
                )
                for _ in range(TEST_SAMPLE_SIZE):
                    x: bytes = os.urandom(n)
                    y: List[int] = decode_bytes_to_polynomial_coefficients(
                        b=x,
                        log2_bias=next_secpar,
                        modulus=params.modulus,
                        degree=params.degree,
                        weight_bound=num_coefs,
                        norm_bound=bound,
                    )
                    y_as_poly: Poly = Poly(
                        modulus=params.modulus,
                        degree=params.degree,
                        root=params.root,
                        inv_root=params.inv_root,
                        root_order=params.root_order,
                        coefficients=y,
                    )
                    assert y_as_poly.norm(p="infty") <= bound
                    assert y_as_poly.weight() <= num_coefs


def test_decode_bytes_to_polynomial_coefficient_redux():
    modulus: int = 65537
    degree: int = 1024
    root_order: int = 2 * degree
    root: int = find_primitive_root(modulus=modulus, root_order=root_order)
    inv_root: int = pow(root, modulus - 2, modulus)
    norm_bound: int = 1000
    weight_bound: int = 100
    seed: int = TEST_SEED

    # First, we will show that the bytestring of all zero-bytes decodes to
    # [-1, -1, -1, ..., -1, -1, 0, 0, ..., 0,  0] before shuffling
    # [0,  -1, -1, ..., -1, -1, 0, 0, ..., 0, -1] after first step
    # [0,  -1, -1, ..., -1, -1, 0, 0, ..., 0, -1] after second step
    # ...
    # [0,  -1, -1, ..., -1, -1, 0, 0, ..., 0, -1] after degree - weight_bound + 1 steps
    # [-1, -1, -1, ..., -1, 0, 0, 0, ...,  0, -1] after degree - weight_bound steps
    # then none of the remaining steps do anything
    # result =  -1 - X - X**2 - ... - X**(weight - 2) - X**(degree - 1)
    expected_coefs: List[int] = [0 for _ in range(degree)]
    for i in range(1, weight_bound):
        expected_coefs[i] = -1
    expected_coefs[-1] = -1

    zero_bytestring: bytes = int("0" * weight_bound, 2).to_bytes(
        byteorder="big", length=ceil(weight_bound / 8)
    )
    log2_bias: int = 256
    bytes_per_coef: int = ceil((log2(norm_bound) + 1 + log2_bias) / 8)
    zero_bytestring += (0).to_bytes(
        byteorder="big", length=bytes_per_coef
    ) * weight_bound
    bytes_per_shuffle_step: int = ceil((log2(degree) + log2_bias) / 8)
    zero_bytestring += (0).to_bytes(
        byteorder="big", length=bytes_per_shuffle_step
    ) * degree

    f: List[int] = decode_bytes_to_polynomial_coefficients(
        b=zero_bytestring,
        log2_bias=log2_bias,
        modulus=modulus,
        degree=degree,
        norm_bound=norm_bound,
        weight_bound=weight_bound,
    )
    assert f == expected_coefs

    # Next, we will show that the bytestring of all 1-bytes decodes to
    # [2, 2, 2, ..., 2, 2, 0, 0, ..., 0, 0] before shuffling
    # [2, 0, 2, ..., 2, 2, 0, 0, ..., 0, 2] after one step
    # [2, 0, 2, ..., 2, 2, 0, 0, ..., 0, 2] after two steps
    # ...
    # [2, 0, 2, ..., 2, 2, 0, 0, ..., 0, 2] after degree - weight_bound + 1 steps
    # [2, 2, 2, ..., 2, 0, 0, 0, ..., 0, 2] after degree - weight_bound steps
    # then none of the remaining steps do anything
    # result = 2 + 2X + 2X**2 + ... + 2X**(weight - 2) + 2X**(degree - 1)
    expected_coefs: List[int] = [0 for _ in range(degree)]
    for i in range(weight_bound):
        expected_coefs[i] = 2
    expected_coefs[1] = 0
    expected_coefs[-1] = 2

    # one_bytestring: bytes = (1).to_bytes(byteorder='big', length=1) * ceil(weight_bound/8)
    one_bytestring: bytes = int("1" * weight_bound, 2).to_bytes(
        byteorder="big", length=ceil(weight_bound / 8)
    )
    bytes_per_coef: int = ceil((log2(norm_bound) + 1 + log2_bias) / 8)
    one_bytestring += (1).to_bytes(
        byteorder="big", length=bytes_per_coef
    ) * weight_bound
    bytes_per_shuffle_step: int = ceil((log2(degree) + log2_bias) / 8)
    one_bytestring += (1).to_bytes(
        byteorder="big", length=bytes_per_shuffle_step
    ) * degree

    f: Poly = decode_bytes_to_polynomial_coefficients(
        b=one_bytestring,
        log2_bias=log2_bias,
        modulus=modulus,
        degree=degree,
        norm_bound=norm_bound,
        weight_bound=weight_bound,
    )
    assert f == expected_coefs


def test_parse_challenge(mocker):
    for next_secpar in [128, 256]:
        # Setup
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)
        num_coefs: int = max(0, min(params.degree, params.omega_ch))
        bound: int = max(0, min(params.modulus // 2, params.beta_ch))
        bytes_per_coefficient: int = ceil((log2(bound) + 1 + next_secpar) / 8)
        bytes_per_index: int = ceil((log2(params.degree) + next_secpar) / 8)
        bytes_for_signums: int = ceil(params.omega_ch / 8)
        total_bytes: int = (
            bytes_for_signums
            + bytes_per_coefficient * num_coefs
            + params.degree * bytes_per_index
        )

        one_poly_coefs: List[int] = [1] + [0 for _ in range(params.degree - 1)]
        one_poly: Poly = Poly(
            modulus=params.modulus,
            degree=params.degree,
            root=params.root,
            inv_root=params.inv_root,
            root_order=params.root_order,
            coefficients=one_poly_coefs,
        )
        one_poly_hat: PolyNTT = transform(one_poly)

        inv_one_poly_hat: Poly = transform(one_poly_hat)
        assert inv_one_poly_hat == one_poly

        # Pick some random input
        b: bytes = os.urandom(
            total_bytes
        )  # NOT SAFE FOR PRODUCTION but suitable for testing

        # Mock the decode_bytes_to_polynomial_coefficient_representation function
        mock_decode_bytes_to_polynomial_coefficients = mocker.patch(
            "fusion.fusion.decode_bytes_to_polynomial_coefficients"
        )
        mock_decode_bytes_to_polynomial_coefficients.return_value = one_poly_coefs
        observed_output: PolyNTT = parse_challenge(params=params, b=b)
        assert one_poly_hat == observed_output

        inv_observed_output: Poly = transform(observed_output)
        assert inv_observed_output == one_poly


def test_hash_ch_mocked(mocker):
    for next_secpar in [128, 256]:
        # Correctness depends on correctness of hash_message_to_int and hash_vk_and_int_to_bytes and parse_challenge
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)
        keys = keygen(params, seed=TEST_SEED + 1)
        otsk, otvk = keys
        msg = "my_message"
        i = hash_message_to_int(params=params, message=msg)

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
        b = hash_vk_and_int_to_bytes(params=params, key=otvk, i=i, n=n)
        assert len(b) >= n

        one_poly_coefs: List[int] = [1] + [0 for _ in range(params.degree - 1)]
        one_poly: Poly = Poly(
            modulus=params.modulus,
            degree=params.degree,
            root=params.root,
            inv_root=params.inv_root,
            root_order=params.root_order,
            coefficients=one_poly_coefs,
        )
        one_poly_hat: PolyNTT = transform(one_poly)

        # Pick some random input
        b: bytes = os.urandom(n)  # NOT SAFE FOR PRODUCTION but suitable for testing

        # Mock the decode_bytes_to_polynomial_coefficient_representation function
        mock_decode_bytes_to_polynomial_coefficients = mocker.patch(
            "fusion.fusion.decode_bytes_to_polynomial_coefficients"
        )
        mock_decode_bytes_to_polynomial_coefficients.return_value = one_poly_coefs
        ch = parse_challenge(params=params, b=b)
        assert ch == one_poly_hat
        expected_result = SignatureChallenge(c_hat=one_poly_hat)

        mock_decode_bytes_to_polynomial_coefficients = mocker.patch(
            "fusion.fusion.decode_bytes_to_polynomial_coefficients"
        )
        mock_decode_bytes_to_polynomial_coefficients.return_value = one_poly_coefs
        observed_result = hash_ch(params=params, key=otvk, message=msg)
        assert expected_result == observed_result


def test_hash_ch():
    for next_secpar in [128, 256]:
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)
        keys: OneTimeKeyTuple = keygen(params=params, seed=TEST_SEED)
        sk: OneTimeSigningKey
        vk: OneTimeVerificationKey
        sk, vk = keys
        message: str = "Hello, world!"
        ct = 0
        while ct < TEST_SAMPLE_SIZE:
            ch: SignatureChallenge = hash_ch(params=params, key=vk, message=message)
            assert isinstance(ch, SignatureChallenge)
            assert isinstance(ch.c_hat, PolyNTT)
            assert ch.c_hat.degree == params.degree
            assert ch.c_hat.modulus == params.modulus
            assert ch.c_hat.root == params.root
            assert ch.c_hat.inv_root == params.inv_root
            assert ch.c_hat.root_order == params.root_order
            assert len(ch.c_hat.values) == params.degree

            c: Poly = transform(ch.c_hat)
            assert isinstance(c, Poly)
            assert c.degree == params.degree
            assert c.modulus == params.modulus
            assert c.root == params.root
            assert c.inv_root == params.inv_root
            assert c.root_order == params.root_order
            assert len(c.coefficients) == params.degree
            assert c.norm(p="infty") <= params.beta_ch
            assert c.weight() <= params.omega_ch

            ct += 1


def test_sign():
    for next_secpar in [128, 256]:
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)
        omega_v_prime: int = min(params.degree, params.omega_sk * (1 + params.omega_ch))
        beta_v_prime: int = params.beta_sk * (
            1 + min(params.degree, params.omega_sk, params.omega_ch) * params.beta_ch
        )

        keys: OneTimeKeyTuple = keygen(params=params, seed=TEST_SEED)
        sk: OneTimeSigningKey
        vk: OneTimeVerificationKey
        sk, vk = keys
        message: str = "Hello, world!"
        ch: SignatureChallenge = hash_ch(params=params, key=vk, message=message)
        sig: Signature = sign(params=params, key=keys, message=message)
        assert isinstance(sig, Signature)
        assert isinstance(sig.signature_hat, GeneralMatrix)
        assert len(sig.signature_hat.matrix) == params.num_rows_sk
        assert len(sig.signature_hat.matrix[0]) == params.num_cols_sk
        assert all(
            isinstance(f, PolyNTT) for row in sig.signature_hat.matrix for f in row
        )
        assert all(
            len(f.values) == params.degree
            for row in sig.signature_hat.matrix
            for f in row
        )
        target: GeneralMatrix = vk.left_vk_hat * ch.c_hat + vk.right_vk_hat
        observed: GeneralMatrix = params.public_challenge * sig.signature_hat
        assert target == observed

        inv_sig_hat: GeneralMatrix = GeneralMatrix(
            matrix=[[transform(f) for f in row] for row in sig.signature_hat.matrix]
        )
        sig_wt: int = inv_sig_hat.weight()
        assert sig_wt <= omega_v_prime
        sig_norm: int = inv_sig_hat.norm(p="infty")
        assert sig_norm <= beta_v_prime

        # omega_v: int = min(params.degree, params.capacity * params.omega_ag * omega_v_prime)
        # beta_v: int = params.capacity * min(params.degree, params.omega_ag, omega_v_prime) * params.beta_ag * beta_v_prime


def test_aggregation_coefficient_class():
    x: AggregationCoefficient = AggregationCoefficient(alpha_hat="Hello world")
    assert isinstance(x, AggregationCoefficient)
    assert x.alpha_hat == "Hello world"
    assert x.__str__() == f"AggregationCoefficient(alpha_hat={'Hello world'})"
    assert x.__str__() == x.__repr__()


def test_hash_vks_and_ints_and_challs_to_bytes():
    pass


def test_decode_bytes_to_agg_coefs():
    pass


def test_hash_ag():
    pass


def test_aggregate():
    pass


# @pytest.mark.skip()
def test_one_sig():
    for next_secpar in [128, 256]:
        # Setup
        params: Params = fusion_setup(secpar=128, seed=TEST_SEED)

        # Keygen
        otk: OneTimeKeyTuple = keygen(params=params, seed=TEST_SEED + 1)
        sk: OneTimeSigningKey = otk[0]
        seed: int = sk.seed
        sk_left_sk_hat: GeneralMatrix = sk.left_sk_hat
        sk_right_sk_hat: GeneralMatrix = sk.right_sk_hat
        vk: OneTimeVerificationKey = otk[1]
        assert params.public_challenge * sk_left_sk_hat == vk.left_vk_hat
        assert params.public_challenge * sk_right_sk_hat == vk.right_vk_hat

        # Sign
        msg: str = "Hello World"
        ch: SignatureChallenge = hash_ch(params=params, key=vk, message=msg)
        sig: Signature = sign(params=params, key=otk, message=msg)
        assert isinstance(sig, Signature)
        assert isinstance(sig.signature_hat, GeneralMatrix)
        assert (
            params.public_challenge * sig.signature_hat
            == vk.left_vk_hat * ch.c_hat + vk.right_vk_hat
        )

        # Aggregate
        alpha_hats: List[AggregationCoefficient] = hash_ag(
            params=params, keys=[vk], messages=[msg]
        )
        agg_sig: Signature = aggregate(
            params=params, keys=[vk], messages=[msg], signatures=[sig]
        )
        assert agg_sig.signature_hat == sig.signature_hat * alpha_hats[0].alpha_hat
        assert (
            params.public_challenge * agg_sig.signature_hat
            == (vk.left_vk_hat * ch.c_hat + vk.right_vk_hat) * alpha_hats[0].alpha_hat
        )

        # Verify
        verification_bit: bool
        explanation: str
        verification_bit, explanation = verify(
            params=params, keys=[vk], messages=[msg], aggregate_signature=agg_sig
        )
        assert explanation == ""
        assert verification_bit


# @pytest.mark.skip()
def test_many_sigs():
    for next_secpar in [128, 256]:
        # Setup
        params: Params = fusion_setup(secpar=next_secpar, seed=TEST_SEED)

        for num_keys in range(1, 5):
            # Keygen
            otks: List[OneTimeKeyTuple] = [
                keygen(params=params, seed=TEST_SEED + 1) for _ in range(num_keys)
            ]

            sks: List[OneTimeSigningKey]
            vks: List[OneTimeVerificationKey]
            sks, vks = [otk[0] for otk in otks], [otk[1] for otk in otks]

            seeds: List[int]
            left_sk_hats: List[GeneralMatrix]
            right_sk_hats: List[GeneralMatrix]
            seeds, left_sk_hats, right_sk_hats = (
                [sk.seed for sk in sks],
                [sk.left_sk_hat for sk in sks],
                [sk.right_sk_hat for sk in sks],
            )

            left_vk_hats: List[GeneralMatrix]
            right_vk_hats: List[GeneralMatrix]
            left_vk_hats, right_vk_hats = [vk.left_vk_hat for vk in vks], [
                vk.right_vk_hat for vk in vks
            ]

            for sk_hat, vk_hat in zip(
                left_sk_hats + right_sk_hats, left_vk_hats + right_vk_hats
            ):
                assert params.public_challenge * sk_hat == vk_hat

            msgs: List[str] = ["test_many_sigs_" + str(i) for i in range(num_keys)]
            sigs: List[Signature] = [
                sign(params=params, key=otk, message=msg)
                for otk, msg in zip(otks, msgs)
            ]

            agg_sig: Signature = aggregate(
                params=params, keys=vks, messages=msgs, signatures=sigs
            )
            assert verify(
                params=params, keys=vks, messages=msgs, aggregate_signature=agg_sig
            )[0]

            # changing any coefficient of any entry in agg_sig by any amount should result in a failure.
            modified_agg_sig: Signature = deepcopy(agg_sig)
            i = randrange(len(modified_agg_sig.signature_hat.matrix))
            j = randrange(len(modified_agg_sig.signature_hat.matrix[0]))
            modified_agg_sig.signature_hat.matrix[i][j].values[0] = (
                modified_agg_sig.signature_hat.matrix[i][j].values[0]
                + randrange(1, params.modulus)
            ) % params.modulus
            assert not verify(
                params=params,
                keys=vks,
                messages=msgs,
                aggregate_signature=modified_agg_sig,
            )[0]

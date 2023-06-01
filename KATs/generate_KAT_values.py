import csv
import os
import random
from math import ceil, log2

from fusion.fusion import aggregate
from fusion.fusion import fusion_setup
from fusion.fusion import hash_ag
from fusion.fusion import hash_ch
from fusion.fusion import hash_message_to_int
from fusion.fusion import hash_vk_and_int_to_bytes
from fusion.fusion import hash_vks_and_ints_and_challs_to_bytes
from fusion.fusion import keygen
from fusion.fusion import sign
from fusion.fusion import verify

print()
# directory to save KAT files
dir_name = "KAT_values"

# create the directory if it doesn't exist
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

secpar_values = [128, 256]

# function names and their arguments
functions = {
    "fusion_setup": ["secpar", "random_seed"],
    "keygen": ["params", "random_seed"],
    "hash_message_to_int": ["params", "random_message"],
    "hash_vk_and_int_to_bytes": ["params", "otk", "i", "n"],
    # and so on for the rest of the functions
}

for secpar in secpar_values:
    seed_a = random.randint(0, 2**32 - 1)
    params = fusion_setup(secpar, seed_a)

    file_path = os.path.join(dir_name, f"fusion_setup_KAT_{secpar}.csv")
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([str((secpar, seed_a)), str(params)])

    (
        random_seed_sks,
        random_msgs,
        otks,
        otvks,
        otsks,
        prehashed_msgs,
        sig_chall_bytes,
        decoded_coefs,
        sig_challs,
        sigs,
    ) = ([], [], [], [], [], [], [], [], [], [])
    for i in range(10):
        random_seed_sks += [random.randint(0, 2**32 - 1)]
        random_msgs += [str(i)]
        otks += [keygen(params, random_seed_sks[i])]

        file_path = os.path.join(dir_name, f"fusion_keygen_KAT_{secpar}.csv")
        with open(file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([str((params, random_seed_sks[i])), str(otks[-1])])

        otsks += [otks[i][0]]
        otvks += [otks[i][1]]
        prehashed_msgs += [hash_message_to_int(params, random_msgs[i])]

        file_path = os.path.join(
            dir_name, f"intermediate_hash_message_to_int_KAT_{secpar}.csv"
        )
        with open(file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([str((params, random_msgs[i])), str(prehashed_msgs[-1])])

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
        sig_chall_bytes += [
            hash_vk_and_int_to_bytes(params, otvks[i], prehashed_msgs[i], n)
        ]

        file_path = os.path.join(
            dir_name, f"intermediate_hash_vk_and_int_to_bytes_to_int_KAT_{secpar}.csv"
        )
        with open(file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    str((params, otvks[i], prehashed_msgs[i], n)),
                    str(sig_chall_bytes[-1]),
                ]
            )

        sig_challs += [hash_ch(params, otvks[i], random_msgs[i])]

        file_path = os.path.join(dir_name, f"intermediate_hash_ch_KAT_{secpar}.csv")
        with open(file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [str((params, otvks[i], random_msgs[i])), str(sig_challs[-1])]
            )

        sigs += [sign(params, otks[i], random_msgs[i])]

        file_path = os.path.join(dir_name, f"fusion_sign_KAT_{secpar}.csv")
        with open(file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([str((params, otks[i], prehashed_msgs[i])), str(sigs[-1])])

    agg_coefs_bytes = hash_vks_and_ints_and_challs_to_bytes(
        params, otks, prehashed_msgs, sig_challs
    )

    file_path = os.path.join(
        dir_name, f"intermediate_hash_vks_and_ints_and_challs_to_bytes_KAT_{secpar}.csv"
    )
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [str((params, otks, prehashed_msgs, sig_challs)), str(agg_coefs_bytes)]
        )

    agg_coefs = hash_ag(params, otks, random_msgs)

    file_path = os.path.join(dir_name, f"intermediate_hash_ag_KAT_{secpar}.csv")
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([str((params, otks, random_msgs)), str(agg_coefs)])

    agg_sig = aggregate(params, otvks, random_msgs, sigs)

    file_path = os.path.join(dir_name, f"fusion_aggregate_KAT_{secpar}.csv")
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([str((params, otvks, random_msgs, sigs)), str(agg_sig)])

    assert verify(params, otvks, random_msgs, agg_sig)

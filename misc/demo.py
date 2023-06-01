import random
import string
from typing import List

from fusion.fusion import (
    fusion_setup,
    keygen,
    sign,
    aggregate,
    verify,
    OneTimeKeyTuple,
    Params,
    Signature,
    OneTimeVerificationKey,
)

# >> Set how many N signatures to create and aggregate for the demo
num_signatures: int = 2

# 1. Set up the cryptographic system using a security parameter and a seed
secpar: int = 256
seed: int = 42
a: Params = fusion_setup(secpar, seed)
print(f"Setup completed with security parameter {secpar} and seed {seed}.")

# 2. Generate N one-time key pairs
keys: List[OneTimeKeyTuple] = [keygen(a, seed) for _ in range(2)]
print(f"Generated {len(keys)} key pairs.")

# 3. Sign N messages using the key pairs
messages: List[str] = [
    "".join(random.choices(string.ascii_letters + string.digits, k=20))
    for _ in range(num_signatures)
]
sigs: List[Signature] = [sign(a, key, message) for key, message in zip(keys, messages)]
print(f"Signed {len(messages)} messages.")

# 4. Aggregate signatures from the signed messages
vks: List[OneTimeVerificationKey] = [key[1] for key in keys]  # public keys
agg_sig: Signature = aggregate(a, vks, messages, sigs)
print("Aggregated the signatures.")

# 5. Verify the aggregate signature
result_bit, result_message = verify(a, vks, messages, agg_sig)
if result_bit:
    print("Verification successful!")
else:
    print(f"Verification failed! Reason: {result_message}")

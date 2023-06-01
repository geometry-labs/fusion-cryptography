# Fusion cryptography

Created by [Geometry Labs](https://www.geometrylabs.io) in partnership with [The QRL Foundation](https://qrl.foundation/)

## Introduction 

Fusion signatures are based on lattice cryptography for post-quantum security, and are highly aggregatable. They are based on the 2023 whitepaper [Fusion One-Time Non-Interactively-Aggregatable Digital Signatures From Lattices](https://eprint.iacr.org/2023/303).


Fusion signatures are elegantly simple in both theory and implementation, reducing the risk of implementation errors. As a one-time signature scheme, Fusion can achieve more narrow tightness gaps in security, smaller signatures, and smaller keys. Our analysis permits parameters to be flexibly tailored to given target security levels and aggregation capacities. Fusion signatures have a number of other generally desirable characteristics: avoiding NTRU assumptions by using only the usual short integer solution assumption, avoiding floating-point arithmetic by using 32-bit signed integers, avoiding trapdoors, and avoiding some issues associated with randomization and statelessness.

**Disclaimer**: Fusion algorithms are still undergoing security analysis and this codebase has not been independently audited.

## Installation

`pip install fusion-cryptography`

## Usage

Here is a little demo showing the full lifecycle of a fusion signature. This includes (1) configuring the cryptographic system, (2) generating multiple keypairs, (3) signing multiple messages, (4) aggregating the signatures, and (5) verifying the result.

```python
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
```

## Tests

Install packages in `requirements-tests.txt` and run scripts in `tests` folder with `pytest`.

## Contributors

Brandon Goodell (lead author), Mitchell P. Krawiec-Thayer
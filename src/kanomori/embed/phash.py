"""Perceptual-hash helpers for visual near-duplicate matching.

PostgreSQL stores pHash values in a signed ``bigint``. Image hashes are conceptually unsigned
64-bit integers, so the conversion helpers preserve the bit pattern while fitting the SQL type.
"""

from __future__ import annotations

MASK_64 = (1 << 64) - 1
SIGN_BIT = 1 << 63
U64_SIZE = 1 << 64


def to_signed_bigint(value: int) -> int:
    """Convert an unsigned 64-bit hash value to PostgreSQL signed bigint representation."""
    value &= MASK_64
    if value & SIGN_BIT:
        return value - U64_SIZE
    return value


def to_u64(value: int) -> int:
    """Return the unsigned 64-bit bit pattern for a signed or unsigned hash value."""
    return value & MASK_64


def hamming_distance(left: int, right: int) -> int:
    """Count differing bits between two signed-bigint-backed 64-bit hashes."""
    return (to_u64(left) ^ to_u64(right)).bit_count()


def compute_phash(image) -> int:
    """Compute a 64-bit pHash for a PIL image and return signed-bigint representation."""
    import imagehash

    return to_signed_bigint(int(str(imagehash.phash(image)), 16))

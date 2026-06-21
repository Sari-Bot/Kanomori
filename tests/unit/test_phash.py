from kanomori.embed.phash import hamming_distance, to_signed_bigint, to_u64


def test_phash_round_trips_unsigned_values_through_signed_bigint() -> None:
    high_bit_hash = 0x8000000000000001

    signed = to_signed_bigint(high_bit_hash)

    assert signed == -9223372036854775807
    assert to_u64(signed) == high_bit_hash


def test_hamming_distance_treats_signed_values_as_64_bit_hashes() -> None:
    left = to_signed_bigint(0x8000000000000000)
    right = to_signed_bigint(0x8000000000000003)

    assert hamming_distance(left, right) == 2

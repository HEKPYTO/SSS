"""prng.py — Windows-UCRT LCG, verbatim ssffs.cpp:86-98"""

_g = 1


def ss_srand(seed: int) -> None:
    global _g
    _g = seed & 0xFFFFFFFF


def ss_rand() -> int:
    global _g
    _g = (_g * 214013 + 2531011) & 0xFFFFFFFF
    return (_g >> 16) & 0x7FFF


SS_RAND_MAX = 32767


def frand() -> float:
    return ss_rand() / (SS_RAND_MAX + 1.0)


def demo() -> None:
    ss_srand(1)
    first5 = [ss_rand() for _ in range(5)]
    assert first5 == [41, 18467, 6334, 26500, 19169], f"PRNG mismatch {first5}"
    print("prng: OK", first5)


if __name__ == "__main__":
    demo()

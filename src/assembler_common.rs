pub const fn is_darwin() -> bool {
    if cfg!(target_os = "macos") {
        true
    } else {
        false
    }
}

pub const fn is_ios() -> bool {
    if cfg!(target_os = "ios") {
        true
    } else {
        false
    }
}

#[macro_export]
macro_rules! is_int {
    ($bits: expr, $t: expr) => {{
        let shift = std::mem::size_of_val(&$t) * 8 - $bits;
        (($t << shift) >> shift) == $t
    }};
}

pub const fn is_int9(value: i32) -> bool {
    value == ((value << 23) >> 23)
}

pub const fn is_uint12(value: isize) -> bool {
    (value & !0xfff) == 0
}

pub const fn is_valid_scaled_uimm12<const DATASIZE: i32>(offset: i32) -> bool {
    let max_pimm = 4095 * (DATASIZE / 8);

    if offset < 8 {
        false
    } else if offset > max_pimm {
        false
    } else if (offset & ((DATASIZE / 8) - 1)) != 0 {
        false
    } else {
        true
    }
}

pub const fn is_valid_signed_imm9(value: i32) -> bool {
    is_int9(value)
}

pub const fn is_valid_signed_imm7(value: i32, alignment_shift_amount: i32) -> bool {
    let disallowed_high_bits = 32 - 7;
    let shifted_value = value >> alignment_shift_amount;
    let fits_in_7_bits =
        ((shifted_value << disallowed_high_bits) >> disallowed_high_bits) == shifted_value;
    let has_correct_alignment = value == (shifted_value << alignment_shift_amount);

    fits_in_7_bits && has_correct_alignment
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ARM64LogicalImmediate {
    pub value: i32,
}

impl ARM64LogicalImmediate {
    pub fn create64(value: u64) -> Self {
        if value == 0 || !value == 0 {
            return Self {
                value: Self::INVALID_LOGICAL_IMMEDIATE,
            };
        }

        let mut hsb = 0;
        let mut lsb = 0;
        let mut inverted = false;

        if Self::find_bit_range::<64>(value, &mut hsb, &mut lsb, &mut inverted) {
            return Self {
                value: Self::encode_logical_immediate::<64>(hsb, lsb, inverted),
            };
        }

        if value as u32 == (value >> 32) as u32 {
            return Self::create32(value as _);
        }

        Self {
            value: Self::INVALID_LOGICAL_IMMEDIATE,
        }
    }

    pub fn create32(mut value: u32) -> ARM64LogicalImmediate {
        if value == 0 || !value == 0 {
            return Self {
                value: Self::INVALID_LOGICAL_IMMEDIATE,
            };
        }

        let mut hsb = 0;
        let mut lsb = 0;
        let mut inverted = false;

        if Self::find_bit_range::<32>(value as _, &mut hsb, &mut lsb, &mut inverted) {
            return Self {
                value: Self::encode_logical_immediate::<32>(hsb, lsb, inverted),
            };
        }

        if (value & 0xffff) != (value >> 16) {
            return Self {
                value: Self::INVALID_LOGICAL_IMMEDIATE,
            };
        }

        value &= 0xffff;

        if Self::find_bit_range::<16>(value as _, &mut hsb, &mut lsb, &mut inverted) {
            return Self {
                value: Self::encode_logical_immediate::<16>(hsb, lsb, inverted),
            };
        }

        if (value & 0xff) != (value >> 8) {
            return Self {
                value: Self::INVALID_LOGICAL_IMMEDIATE,
            };
        }

        value &= 0xff;

        if Self::find_bit_range::<8>(value as _, &mut hsb, &mut lsb, &mut inverted) {
            return Self {
                value: Self::encode_logical_immediate::<8>(hsb, lsb, inverted),
            };
        }

        if (value & 0xf) != (value >> 4) {
            return Self {
                value: Self::INVALID_LOGICAL_IMMEDIATE,
            };
        }

        value &= 0xf;

        if Self::find_bit_range::<4>(value as _, &mut hsb, &mut lsb, &mut inverted) {
            return Self {
                value: Self::encode_logical_immediate::<4>(hsb, lsb, inverted),
            };
        }

        if (value & 0x3) != (value >> 2) {
            return Self {
                value: Self::INVALID_LOGICAL_IMMEDIATE,
            };
        }

        value &= 0x3;

        if Self::find_bit_range::<2>(value as _, &mut hsb, &mut lsb, &mut inverted) {
            return Self {
                value: Self::encode_logical_immediate::<2>(hsb, lsb, inverted),
            };
        }

        Self {
            value: Self::INVALID_LOGICAL_IMMEDIATE,
        }
    }

    pub const fn is_64bit(self) -> bool {
        (self.value & (1 << 12)) != 0
    }

    pub const fn value(self) -> i32 {
        self.value
    }

    pub const fn is_valid(self) -> bool {
        self.value != Self::INVALID_LOGICAL_IMMEDIATE
    }

    pub const INVALID_LOGICAL_IMMEDIATE: i32 = -1;

    /// This function takes a value and a bit width, where value obeys the following constraints:
    ///   * bits outside of the width of the value must be zero.
    ///   * bits within the width of value must neither be all clear or all set.
    /// The input is inspected to detect values that consist of either two or three contiguous
    /// ranges of bits. The output range hsb..lsb will describe the second range of the value.
    /// if the range is set, inverted will be false, and if the range is clear, inverted will
    /// be true. For example (with width 8):
    ///   00001111 = hsb:3, lsb:0, inverted:false
    ///   11110000 = hsb:3, lsb:0, inverted:true
    ///   00111100 = hsb:5, lsb:2, inverted:false
    ///   11000011 = hsb:5, lsb:2, inverted:true
    pub fn find_bit_range<const WIDTH: usize>(
        mut value: u64,
        hsb: &mut usize,
        lsb: &mut usize,
        inverted: &mut bool,
    ) -> bool {
        let msb = 1 << (WIDTH as u64 - 1);
        *inverted = (value & msb) != 0;
        if *inverted {
            value ^= Self::mask(WIDTH - 1);
        }

        *hsb = Self::highest_set_bit(value);

        value ^= Self::mask(*hsb);

        if value == 0 {
            *lsb = 0;
            return true;
        }

        *lsb = Self::highest_set_bit(value);
        value ^= Self::mask(*lsb);

        if value == 0 {
            *lsb += 1;
            return true;
        }
        false
    }

    /// Encodes the set of immN:immr:imms fields found in a logical immediate.
    pub fn encode_logical_immediate<const WIDTH: usize>(
        hsb: usize,
        lsb: usize,
        inverted: bool,
    ) -> i32 {
        let mut immn = 0;
        let mut imms = 0;
        let immr;

        // For 64-bit values this is easy - just set immN to true, and imms just
        // contains the bit number of the highest set bit of the set range. For
        // values with narrower widths, these are encoded by a leading set of
        // one bits, followed by a zero bit, followed by the remaining set of bits
        // being the high bit of the range. For a 32-bit immediate there are no
        // leading one bits, just a zero followed by a five bit number. For a
        // 16-bit immediate there is one one bit, a zero bit, and then a four bit
        // bit-position, etc.
        if WIDTH == 64 {
            immn = 1;
        } else {
            imms = 63 & !(WIDTH + WIDTH - 1);
        }

        if inverted {
            immr = (WIDTH - 1) - hsb;
            imms |= (WIDTH - ((hsb - lsb) + 1)) - 1;
        } else {
            immr = (WIDTH - lsb) & (WIDTH - 1);
            imms |= hsb - lsb;
        }
        (immn << 12 | immr << 6 | imms) as i32
    }

    /// Generate a mask with bits in the range hsb..0 set, for example:
    ///   hsb:63 = 0xffffffffffffffff
    ///   hsb:42 = 0x000007ffffffffff
    ///   hsb: 0 = 0x0000000000000001
    pub const fn mask(hsb: usize) -> u64 {
        0xffffffffffffffff >> (63 - hsb as u64)
    }

    pub fn partial_hsb<const N: usize>(value: &mut u64, result: &mut usize) {
        if (*value & (0xffffffffffffffff << N as u64)) != 0 {
            *result += N;
            *value >>= N as u64;
        }
    }
    /// Find the bit number of the highest bit set in a non-zero value, for example:
    ///   0x8080808080808080 = hsb:63
    ///   0x0000000000000001 = hsb: 0
    ///   0x000007ffffe00000 = hsb:42
    pub fn highest_set_bit(mut value: u64) -> usize {
        let mut hsb = 0;

        Self::partial_hsb::<32>(&mut value, &mut hsb);
        Self::partial_hsb::<16>(&mut value, &mut hsb);
        Self::partial_hsb::<8>(&mut value, &mut hsb);
        Self::partial_hsb::<4>(&mut value, &mut hsb);
        Self::partial_hsb::<2>(&mut value, &mut hsb);
        Self::partial_hsb::<1>(&mut value, &mut hsb);

        hsb
    }
}

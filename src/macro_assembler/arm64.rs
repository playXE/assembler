use std::ops::{Deref, DerefMut};

use crate::{
    arm64::*,
    assembler_common::{
        is_uint12, is_valid_scaled_uimm12, is_valid_scaled_uimm12_2, is_valid_signed_imm9,
        ARM64LogicalImmediate,
    },
};

use super::{
    Address, BaseIndex, CachedTempRegister, Call, DataLabel32, DataLabelCompact, Extend,
    ExtendedAddress, Jump, MacroAssembler, PostIndexAddress, PreIndexAddress, Scale,
};

macro_rules! has_one_bit_set {
    ($value: expr) => {
        (($value - 1) & $value) != 0 && $value != 0
    };
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
#[repr(i32)]
pub enum RelationalCondition {
    Equal = Condition::EQ as i32,
    NotEqual = Condition::NE as i32,
    Above = Condition::HI as i32,
    AboveOrEqual = Condition::HS as i32,
    Below = Condition::LO as i32,
    BelowOrEqual = Condition::LS as i32,
    GreaterThan = Condition::GT as i32,
    GreaterThanOrEqual = Condition::GE as i32,
    LessThan = Condition::LT as i32,
    LessThanOrEqual = Condition::LE as i32,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
#[repr(i32)]
pub enum ResultCondition {
    Overflow = Condition::VS as i32,
    Signed = Condition::MI as i32,
    PositiveOrZero = Condition::PL as i32,
    Zero = Condition::EQ as i32,
    NonZero = Condition::NE as i32,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
#[repr(i32)]
pub enum ZeroCondition {
    IsZero = Condition::EQ as i32,
    IsNonZero = Condition::NE as i32,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
#[repr(i32)]
pub enum DoubleCondition {
    EqualAndOrdered = Condition::EQ as i32,
    NotEqualAndOrdered = Condition::VC as i32,
    GreaterThanAndOrdered = Condition::GT as i32,
    GreaterThanOrEqualAndOrdered = Condition::GE as i32,
    LessThanAndOrdered = Condition::LO as i32,
    LessThanOrEqualAndOrdered = Condition::LS as i32,

    EqualOrUnordered = Condition::VS as i32,
    NotEqualOrUnordered = Condition::NE as i32,
    GreaterThanOrUnordered = Condition::HI as i32,
    GreaterThanOrEqualOrUnordered = Condition::HS as i32,
    LessThanOrUnordered = Condition::LT as i32,
    LessThanOrEqualOrUnordered = Condition::LE as i32,
}

#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Debug, Hash)]
pub struct FR(pub RegisterId);
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Debug, Hash)]
pub struct GR(pub RegisterId);

pub const STACK_POINTER_REGISTER: GR = GR(sp);
pub const FRAME_POINTER_REGISTER: GR = GR(fp);
pub const LINK_REGISTER: GR = GR(lr);

pub struct MacroAssemblerARM64 {
    base: ARM64Assembler,
    make_jump_patchable: bool,
    temp_registers_valid_bits: usize,
    allow_scratch_register: bool,
    cached_memory_temp_register: CachedTempRegister,
    cached_data_memory_temp_register: CachedTempRegister,
}

impl Deref for MacroAssemblerARM64 {
    type Target = ARM64Assembler;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for MacroAssemblerARM64 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl MacroAssembler for MacroAssemblerARM64 {
    type AssemblerType = ARM64Assembler;

    fn invalidate_all_temp_registers(&mut self) {
        self.temp_registers_valid_bits = 0;
    }
    fn is_temp_register_valid(&self, r: usize) -> bool {
        (self.temp_registers_valid_bits & r) != 0
    }

    fn clear_temp_register_valid(&mut self, r: usize) {
        self.temp_registers_valid_bits &= !r;
    }

    fn set_temp_register_valid(&mut self, r: usize) {
        self.temp_registers_valid_bits |= r;
    }

    fn first_fp_register() -> RegisterId {
        q0
    }

    fn last_fp_register() -> RegisterId {
        q31
    }

    fn first_register() -> RegisterId {
        x0
    }

    fn last_register() -> RegisterId {
        sp
    }

    fn first_sp_register() -> RegisterId {
        pc
    }

    fn last_sp_register() -> RegisterId {
        fpsr
    }

    fn number_of_registers() -> usize {
        (Self::last_register() - Self::first_register() + 1) as _
    }

    fn number_of_fp_registers() -> usize {
        (Self::last_fp_register() - Self::first_fp_register() + 1) as _
    }

    fn number_of_sp_registers() -> usize {
        (Self::last_sp_register() - Self::first_sp_register() + 1) as _
    }

    fn link_call(code: *mut u8, from: Call, function: *mut u8) {
        unsafe {
            if !from.is_flag_set(super::CallFlags::Near) {
                ARM64Assembler::link_pointer(code, from.label, function);
            } else if from.is_flag_set(super::CallFlags::Tail) {
                ARM64Assembler::link_jump(code, from.label, function);
            } else {
                ARM64Assembler::link_call(code, from.label, function)
            }
        }
    }

    fn link_jump(code: *mut u8, from: crate::assembler_buffer::AssemblerLabel, to: *mut u8) {
        unsafe {
            ARM64Assembler::link_jump(code, from, to);
        }
    }

    fn link_pointer(
        code: *mut u8,
        at: crate::assembler_buffer::AssemblerLabel,
        value_ptr: *mut u8,
    ) {
        unsafe {
            ARM64Assembler::link_pointer(code, at, value_ptr);
        }
    }

    fn compute_jump_type(typ: JumpType, from: *const u8, to: *const u8) -> JumpLinkType {
        unsafe { compute_jump_type(typ, from, to) }
    }

    fn jump_size_delta(typ: JumpType, link: JumpLinkType) -> i32 {
        jump_size_delta(typ, link)
    }

    fn can_compact(typ: JumpType) -> bool {
        can_compact(typ)
    }

    fn get_call_return_offset(label: crate::assembler_buffer::AssemblerLabel) -> usize {
        ARM64Assembler::get_call_return_offset(label) as _
    }

    fn raw_label(&mut self) -> crate::assembler_buffer::AssemblerLabel {
        self.base.label()
    }

    fn label(&mut self) -> super::Label {
        super::Label::new(self)
    }

    fn align(&mut self) -> super::Label {
        self.base.align(16);
        self.label()
    }
    fn link_jump_(&mut self, j: super::Jump) {
        self.base.link_jump_(j);
    }

    fn link_to(&mut self, j: super::Jump, label: super::Label) {
        self.base.link_to(j, label)
    }

    type GR = GR;
    type FR = FR;
}

pub const DATA_TEMP_REGISTER: GR = GR(ip0);
pub const MEMORY_TEMP_REGISTER: GR = GR(ip1);

impl MacroAssemblerARM64 {
    pub const MASK_HALF_WORD_0: i64 = 0xffff;
    pub const MASK_HALF_WORD_1: i64 = 0xffff0000;
    pub const MASK_UPPER_WORD: i64 = 0xffffffff00000000u64 as i64;

    pub const NEAR_JUMP_RANGE: usize = 128 * 1024 * 1024;
    pub const INSTRUCTION_SIZE: usize = 4;
    pub const REPATCH_OFFSET_CALL_TO_POINTER: isize = -((MAX_POINTER_BITS as isize / 16 + 1) * 4);

    pub fn new() -> Self {
        Self {
            base: ARM64Assembler::new(),
            make_jump_patchable: false,
            allow_scratch_register: true,
            temp_registers_valid_bits: 0,
            cached_data_memory_temp_register: CachedTempRegister::new(DATA_TEMP_REGISTER.0),
            cached_memory_temp_register: CachedTempRegister::new(MEMORY_TEMP_REGISTER.0),
        }
    }

    pub fn data_temp_memory_register_mut(&mut self) -> &mut CachedTempRegister {
        &mut self.cached_data_memory_temp_register
    }

    pub fn data_temp_memory_register(&self) -> &CachedTempRegister {
        &self.cached_data_memory_temp_register
    }

    pub fn memory_temp_register_mut(&mut self) -> &mut CachedTempRegister {
        &mut self.cached_memory_temp_register
    }

    pub fn memory_temp_register(&self) -> &CachedTempRegister {
        &self.cached_memory_temp_register
    }

    pub fn get_cached_memory_temp_register_and_invalidate(&mut self) -> GR {
        let mut r = *self.memory_temp_register();
        let rr = r.register_id_invalidate(self);
        GR(rr)
    }

    pub fn get_cached_data_memory_temp_register_and_invalidate(&mut self) -> GR {
        let mut r = *self.data_temp_memory_register();
        let rr = r.register_id_invalidate(self);
        GR(rr)
    }

    pub fn scratch_register(&mut self) -> GR {
        self.get_cached_data_memory_temp_register_and_invalidate()
    }

    pub fn add32(&mut self, mut a: GR, mut b: GR, dest: GR) {
        if b.0 == sp {
            std::mem::swap(&mut a, &mut b);
        }

        self.add(32, SetFlags::DontSet, dest.0, a.0, b.0);
    }

    pub fn add32_rr(&mut self, src: GR, dest: GR) {
        if src.0 == sp {
            self.add(32, SetFlags::DontSet, dest.0, src.0, dest.0);
        } else {
            self.add(32, SetFlags::DontSet, dest.0, dest.0, src.0);
        }
    }

    pub fn add32_imm(&mut self, imm: i32, src: GR, dest: GR) {
        if is_uint12(imm as _) {
            self.base
                .add_imm12(32, SetFlags::DontSet, dest.0, src.0, imm, 0);
        } else if is_uint12(-imm as isize) {
            self.base
                .sub_imm12(32, SetFlags::DontSet, dest.0, src.0, -imm, 0);
        } else if src != dest {
            self.mov_imm32(imm, dest);
            self.add32_rr(src, dest);
        } else {
            let r = self.get_cached_memory_temp_register_and_invalidate();

            self.mov_imm32(imm, r);
            self.base.add(32, SetFlags::DontSet, dest.0, src.0, r.0);
        }
    }

    pub fn add32_imm_rr(&mut self, imm: i32, dest: GR) {
        self.add32_imm(imm, dest, dest);
    }

    pub fn mov(&mut self, src: GR, dest: GR) {
        if src != dest {
            self.base.mov(64, dest.0, src.0);
        }
    }

    pub fn mov_imm32(&mut self, src: i32, dest: GR) {
        self.move_internal32(src as _, dest)
    }

    pub fn mov_imm64(&mut self, src: i64, dest: GR) {
        self.move_internal64(src as _, dest);
    }

    fn move_internal32(&mut self, imm: u32, dest: GR) {
        const DATASIZE: i32 = 4 * 8;
        const NUMBER_HALF_WORDS: i32 = DATASIZE / 16;
        let mut halfword = [0u16; NUMBER_HALF_WORDS as usize];

        if imm == 0 {
            self.base.movz(DATASIZE, dest.0, 0, 0);
            return;
        }

        if (!imm) == 0 {
            self.base.movn(DATASIZE, dest.0, 0, 0);
            return;
        }

        let logical_imm = ARM64LogicalImmediate::create32(imm);

        if logical_imm.is_valid() {
            self.base.movi(DATASIZE, dest.0, logical_imm.value);
            return;
        }

        let mut zero_or_negate_vote = 0;

        for i in 0..NUMBER_HALF_WORDS as usize {
            halfword[i] = get_half_word32(imm, i as _);

            if halfword[i] == 0 {
                zero_or_negate_vote += 1;
            } else if halfword[i] == 0xffff {
                zero_or_negate_vote -= 1;
            }
        }

        let mut need_to_clear_register = true;

        if zero_or_negate_vote >= 0 {
            for i in 0..NUMBER_HALF_WORDS as usize {
                if halfword[i] != 0 {
                    if need_to_clear_register {
                        self.base.movz(DATASIZE, dest.0, halfword[i], 16 * i as i32);
                        need_to_clear_register = false;
                    } else {
                        self.base.movk(DATASIZE, dest.0, halfword[i], 16 * i as i32);
                    }
                }
            }
        } else {
            for i in 0..NUMBER_HALF_WORDS as usize {
                if halfword[i] != 0xffff {
                    if need_to_clear_register {
                        self.base
                            .movz(DATASIZE, dest.0, !halfword[i], 16 * i as i32);
                        need_to_clear_register = false;
                    } else {
                        self.base.movk(DATASIZE, dest.0, halfword[i], 16 * i as i32);
                    }
                }
            }
        }
    }

    fn move_internal64(&mut self, imm: u64, dest: GR) {
        const DATASIZE: i32 = 8 * 8;
        const NUMBER_HALF_WORDS: i32 = DATASIZE / 16;
        let mut halfword = [0u16; NUMBER_HALF_WORDS as usize];

        if imm == 0 {
            self.base.movz(DATASIZE, dest.0, 0, 0);
            return;
        }

        if (!imm) == 0 {
            self.base.movn(DATASIZE, dest.0, 0, 0);
            return;
        }

        let logical_imm = ARM64LogicalImmediate::create64(imm);

        if logical_imm.is_valid() {
            self.base.movi(DATASIZE, dest.0, logical_imm.value);
            return;
        }

        let mut zero_or_negate_vote = 0;

        for i in 0..NUMBER_HALF_WORDS as usize {
            halfword[i] = get_half_word(imm, i as _);

            if halfword[i] == 0 {
                zero_or_negate_vote += 1;
            } else if halfword[i] == 0xffff {
                zero_or_negate_vote -= 1;
            }
        }

        let mut need_to_clear_register = true;

        if zero_or_negate_vote >= 0 {
            for i in 0..NUMBER_HALF_WORDS as usize {
                if halfword[i] != 0 {
                    if need_to_clear_register {
                        self.base.movz(DATASIZE, dest.0, halfword[i], 16 * i as i32);
                        need_to_clear_register = false;
                    } else {
                        self.base.movk(DATASIZE, dest.0, halfword[i], 16 * i as i32);
                    }
                }
            }
        } else {
            for i in 0..NUMBER_HALF_WORDS as usize {
                if halfword[i] != 0xffff {
                    if need_to_clear_register {
                        self.base
                            .movz(DATASIZE, dest.0, !halfword[i], 16 * i as i32);
                        need_to_clear_register = false;
                    } else {
                        self.base.movk(DATASIZE, dest.0, halfword[i], 16 * i as i32);
                    }
                }
            }
        }
    }

    fn fp_cmp(
        &mut self,
        cond: DoubleCondition,
        left: RegisterId,
        right: RegisterId,
        dest: RegisterId,
        compare: impl FnOnce(FR, FR),
    ) {
        if cond == DoubleCondition::NotEqualAndOrdered {
            self.mov_imm32(0, GR(dest));
            compare(FR(left), FR(right));
            let unordered = self.make_branch_raw(Condition::VS);
            self.base.cset(32, dest, Condition::NE);
            self.link_jump_(unordered);
            return;
        }

        if cond == DoubleCondition::EqualOrUnordered {
            self.mov_imm32(1, GR(dest));
            compare(FR(left), FR(right));
            let unordered = self.make_branch_raw(Condition::VS);
            self.base.cset(32, dest, Condition::EQ);
            self.link_jump_(unordered);
            return;
        }
        compare(FR(left), FR(right));
        self.base
            .cset(32, dest, unsafe { std::mem::transmute(cond) });
    }

    pub fn make_branch_raw(&mut self, cond: Condition) -> Jump {
        self.b_cond(cond, 0);
        let label = self.base.label();
        self.base.nop();

        Jump {
            label,
            typ: if self.make_jump_patchable {
                JumpType::ConditionFixedSize
            } else {
                JumpType::Condition
            },
            cond,
            compare_register: 0,
            bitnumber: 0,
            is64bit: false,
        }
    }

    pub fn make_branch_rel(&mut self, cond: RelationalCondition) -> Jump {
        self.make_branch_raw(unsafe { std::mem::transmute(cond) })
    }

    pub fn make_branch_res(&mut self, cond: ResultCondition) -> Jump {
        self.make_branch_raw(unsafe { std::mem::transmute(cond) })
    }

    pub fn make_branch_fp(&mut self, cond: DoubleCondition) -> Jump {
        self.make_branch_raw(unsafe { std::mem::transmute(cond) })
    }

    pub fn make_compare_and_branch(&mut self, datasize: i32, cond: ZeroCondition, reg: GR) -> Jump {
        if cond == ZeroCondition::IsZero {
            self.base.cbz(datasize, reg.0, 0);
        } else {
            self.base.cbnz(datasize, reg.0, 0);
        }

        let label = self.base.label();
        self.base.nop();
        Jump {
            label,
            typ: if self.make_jump_patchable {
                JumpType::CompareAndBranchFixedSize
            } else {
                JumpType::CompareAndBranch
            },
            cond: unsafe { std::mem::transmute(cond) },
            compare_register: reg.0,
            bitnumber: 0,
            is64bit: false,
        }
    }
    pub fn make_test_bit_and_branch(
        &mut self,
        datasize: i32,
        reg: GR,
        bit: usize,
        cond: ZeroCondition,
    ) -> Jump {
        if cond == ZeroCondition::IsZero {
            self.base.tbz(reg.0, bit as _, 0);
        } else {
            self.base.tbnz(reg.0, bit as _, 0);
        }

        let label = self.base.label();
        self.base.nop();
        Jump {
            label,
            typ: if self.make_jump_patchable {
                JumpType::TestBitFixedSize
            } else {
                JumpType::TestBit
            },
            cond: unsafe { std::mem::transmute(cond) },
            compare_register: reg.0,
            bitnumber: bit,
            is64bit: false,
        }
    }
    pub fn jump_after_fp_cmp(&mut self, cond: DoubleCondition) -> Jump {
        if cond == DoubleCondition::NotEqualAndOrdered {
            let unordered = self.make_branch_raw(Condition::VS);
            let result = self.make_branch_raw(Condition::NE);
            self.link_jump_(unordered);
            return result;
        }
        if cond == DoubleCondition::EqualOrUnordered {
            let unordered = self.make_branch_raw(Condition::VS);
            let not_eq = self.make_branch_raw(Condition::NE);
            self.link_jump_(unordered);

            let result = self.jump();
            self.link_jump_(not_eq);
            return result;
        }
        self.make_branch_fp(cond)
    }

    pub fn jump(&mut self) -> Jump {
        let label = self.base.label();
        self.base.b();
        Jump {
            label,
            typ: if self.make_jump_patchable {
                JumpType::NoConditionFixedSize
            } else {
                JumpType::NoCondition
            },
            cond: Condition::AL,
            bitnumber: 0,
            is64bit: false,
            compare_register: 0,
        }
    }

    fn load_unsigned_immediate(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        pimm: usize,
    ) {
        if datasize == 8 {
            self.base.ldrb_pimm(rt, rn, pimm)
        } else if datasize == 16 {
            self.base.ldrh_pimm(rt, rn, pimm)
        } else {
            self.base.ldr_pimm(datasize, rt, rn, pimm);
        }
    }

    fn load_unscaled_immediate(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        simm: i32,
    ) {
        if datasize == 8 {
            self.base.ldurb(rt, rn, simm)
        } else if datasize == 16 {
            self.base.ldurh(rt, rn, simm);
        } else {
            self.base.ldur(datasize, rt, rn, simm)
        }
    }

    fn store_unsigned_immediate(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        pimm: usize,
    ) {
        self.base.str_pimm(datasize, rt, rn, pimm);
    }

    fn store_unscaled_immediate(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        simm: i32,
    ) {
        self.base.stur(datasize, rt, rn, simm)
    }

    fn move_with_fixed_width(&mut self, imm: i32, dest: RegisterId) {
        self.base.movz(32, dest, get_half_word32(imm as u32, 0), 0);
        self.base.movk(32, dest, get_half_word32(imm as u32, 1), 16);
    }

    fn move_with_fixed_width_ptr(&mut self, imm: isize, dest: RegisterId) {
        self.base.movz(64, dest, get_half_word(imm as _, 0), 0);
        self.base.movk(64, dest, get_half_word(imm as _, 1), 16);
        self.base.movk(64, dest, get_half_word(imm as _, 2), 32);
    }

    pub fn sign_extend_32_to_ptr_with_fixed_width(&mut self, imm: i32, dest: RegisterId) {
        if imm >= 0 {
            self.base.movz(32, dest, get_half_word32(imm as u32, 0), 0);
            self.base.movk(32, dest, get_half_word32(imm as u32, 1), 16);
        } else {
            self.base.movn(32, dest, !get_half_word32(imm as u32, 0), 0);
            self.base.movk(32, dest, get_half_word32(imm as u32, 1), 16);
        }
    }

    fn load(&mut self, datasize: i32, address: *mut u8, dest: GR) {
        let mut cur_contents = 0;
        if self
            .cached_memory_temp_register
            .value(self, &mut cur_contents)
        {
            let address_as_int = address as isize;
            let address_delta = address_as_int - cur_contents;

            if dest == MEMORY_TEMP_REGISTER {
                let mut r = self.cached_memory_temp_register;
                r.invalidate(self);
            }

            if is_int!(32, address_delta) {
                if is_valid_signed_imm9(address_delta as _) {
                    self.load_unscaled_immediate(
                        datasize,
                        dest.0,
                        MEMORY_TEMP_REGISTER.0,
                        address_delta as _,
                    );
                    return;
                }

                if is_valid_scaled_uimm12_2(datasize, address_delta as _) {
                    self.load_unsigned_immediate(
                        datasize,
                        dest.0,
                        MEMORY_TEMP_REGISTER.0,
                        address_delta as _,
                    );
                }
            }

            if (address_as_int & (!Self::MASK_HALF_WORD_0 as isize))
                == (cur_contents as isize & (!Self::MASK_HALF_WORD_0 as isize))
            {
                self.base.movk(
                    64,
                    MEMORY_TEMP_REGISTER.0,
                    (address_as_int & Self::MASK_HALF_WORD_0 as isize) as u16,
                    0,
                );

                let mut r = self.cached_memory_temp_register;
                r.set_value(self, address as _);
                self.cached_memory_temp_register = r;

                if datasize == 16 {
                    self.base.ldrh(dest.0, MEMORY_TEMP_REGISTER.0, zr);
                } else {
                    self.base.ldr(datasize, dest.0, MEMORY_TEMP_REGISTER.0, zr);
                }
                return;
            }
        }

        self.mov_imm64(address as _, MEMORY_TEMP_REGISTER);

        if dest == MEMORY_TEMP_REGISTER {
            let mut r = self.cached_memory_temp_register;
            r.invalidate(self);
        } else {
            let mut r = self.cached_memory_temp_register;
            r.set_value(self, address as _);
            self.cached_memory_temp_register = r;
        }
        if datasize == 16 {
            self.base.ldrh(dest.0, MEMORY_TEMP_REGISTER.0, zr);
        } else {
            self.base.ldr(datasize, dest.0, MEMORY_TEMP_REGISTER.0, zr);
        }
    }

    fn store(&mut self, datasize: i32, src: GR, address: *mut u8) {
        let mut cur_contents = 0;
        if self
            .cached_memory_temp_register
            .value(self, &mut cur_contents)
        {
            let address_as_int = address as isize;
            let address_delta = address_as_int - cur_contents;

            if is_int!(32, address_delta) {
                if is_valid_signed_imm9(address_delta as _) {
                    self.store_unscaled_immediate(
                        datasize,
                        src.0,
                        MEMORY_TEMP_REGISTER.0,
                        address_delta as _,
                    );
                    return;
                }

                if is_valid_scaled_uimm12_2(datasize, address_delta as _) {
                    self.store_unsigned_immediate(
                        datasize,
                        src.0,
                        MEMORY_TEMP_REGISTER.0,
                        address_delta as _,
                    );
                }
            }

            if (address_as_int & (!Self::MASK_HALF_WORD_0 as isize))
                == (cur_contents as isize & (!Self::MASK_HALF_WORD_0 as isize))
            {
                self.base.movk(
                    64,
                    MEMORY_TEMP_REGISTER.0,
                    (address_as_int & Self::MASK_HALF_WORD_0 as isize) as u16,
                    0,
                );

                let mut r = self.cached_memory_temp_register;
                r.set_value(self, address as _);
                self.cached_memory_temp_register = r;

                if datasize == 16 {
                    self.base.strh(src.0, MEMORY_TEMP_REGISTER.0, zr);
                } else {
                    self.base.str(datasize, src.0, MEMORY_TEMP_REGISTER.0, zr);
                }
                return;
            }
        }
    }

    fn try_move_using_cache_register_contents(
        &mut self,
        datasize: i32,
        immediate: isize,
        dest: &mut CachedTempRegister,
    ) -> bool {
        let mut cur_contents = 0;
        if dest.value(self, &mut cur_contents) {
            if cur_contents == immediate {
                return true;
            }

            let logical_imm = if datasize == 64 {
                ARM64LogicalImmediate::create64(immediate as _)
            } else {
                ARM64LogicalImmediate::create32(immediate as _)
            };

            if logical_imm.is_valid() {
                self.base.movi(
                    datasize,
                    dest.register_id_no_invalidate(),
                    logical_imm.value,
                );
                dest.set_value(self, immediate as _);
                return true;
            }

            if (immediate & Self::MASK_UPPER_WORD as isize)
                == (cur_contents & Self::MASK_UPPER_WORD as isize)
            {
                if (immediate & Self::MASK_HALF_WORD_1 as isize)
                    != (cur_contents & Self::MASK_HALF_WORD_1 as isize)
                {
                    self.base.movk(
                        datasize,
                        dest.register_id_no_invalidate(),
                        ((immediate & Self::MASK_HALF_WORD_1 as isize) >> 16) as _,
                        16,
                    );
                }

                if (immediate & Self::MASK_HALF_WORD_0 as isize)
                    != (cur_contents & Self::MASK_HALF_WORD_0 as isize)
                {
                    self.base.movk(
                        datasize,
                        dest.register_id_no_invalidate(),
                        (immediate & Self::MASK_HALF_WORD_0 as isize) as _,
                        0,
                    );
                }

                dest.set_value(self, immediate as _);
                return true;
            }
        }
        false
    }

    fn move_to_cached_reg32(&mut self, imm: i32, dest: &mut CachedTempRegister) {
        if self.try_move_using_cache_register_contents(32, imm as _, dest) {
            return;
        }

        self.move_internal32(imm as _, GR(dest.register_id_no_invalidate()));
        dest.set_value(self, imm as _);
    }

    fn move_to_cached_reg64(&mut self, imm: i64, dest: &mut CachedTempRegister) {
        if self.try_move_using_cache_register_contents(64, imm as _, dest) {
            return;
        }

        self.move_internal64(imm as _, GR(dest.register_id_no_invalidate()));
        dest.set_value(self, imm as _);
    }

    fn try_load_with_offset(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        offset: i32,
    ) -> bool {
        if is_valid_signed_imm9(offset) {
            self.load_unscaled_immediate(datasize, rt, rn, offset);
            return true;
        }

        if is_valid_scaled_uimm12_2(datasize, offset) {
            self.load_unsigned_immediate(datasize, rt, rn, offset as _);
            return true;
        }
        false
    }

    fn try_load_signed_with_offset(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        offset: i32,
    ) -> bool {
        if is_valid_signed_imm9(offset) {
            self.load_signed_addressed_by_unscaled_immediate(datasize, rt, rn, offset);
            return true;
        }

        if is_valid_scaled_uimm12_2(datasize, offset) {
            self.load_signed_addressed_by_unsigned_immediate(datasize, rt, rn, offset as _);
            return true;
        }
        false
    }

    fn try_load_with_offset_fp(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        offset: i32,
    ) -> bool {
        if is_valid_signed_imm9(offset) {
            self.base.ldur_f(datasize, rt, rn, offset);
            return true;
        }

        if is_valid_scaled_uimm12_2(datasize, offset) {
            self.base.ldr_f_pimm(datasize, rt, rn, offset as _);
            return true;
        }

        false
    }

    fn try_store_with_offset(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        offset: i32,
    ) -> bool {
        if is_valid_signed_imm9(offset) {
            self.store_unscaled_immediate(datasize, rt, rn, offset);
            return true;
        }

        if is_valid_scaled_uimm12_2(datasize, offset) {
            self.store_unsigned_immediate(datasize, rt, rn, offset as _);
            return true;
        }
        false
    }

    fn try_store_with_offset_fp(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        offset: i32,
    ) -> bool {
        if is_valid_signed_imm9(offset) {
            self.base.stur_f(datasize, rt, rn, offset);
            return true;
        }

        if is_valid_scaled_uimm12_2(datasize, offset) {
            self.base.str_f_pimm(datasize, rt, rn, offset as _);
            return true;
        }
        false
    }

    fn try_fold_base_and_offset_part(&mut self, address: BaseIndex) -> Option<RegisterId> {
        if address.offset == 0 {
            return Some(address.base);
        }

        if is_uint12(address.offset as _) {
            let tmp = self.get_cached_memory_temp_register_and_invalidate();
            self.base.add_imm12(
                64,
                SetFlags::DontSet,
                tmp.0,
                address.base,
                address.offset as _,
                0,
            );
            return Some(MEMORY_TEMP_REGISTER.0);
        }

        if is_uint12(-address.offset as isize) {
            let tmp = self.get_cached_memory_temp_register_and_invalidate();
            self.base.sub_imm12(
                64,
                SetFlags::DontSet,
                tmp.0,
                address.base,
                -address.offset as _,
                0,
            );
            return Some(MEMORY_TEMP_REGISTER.0);
        }
        None
    }

    fn load_signed_addressed_by_unscaled_immediate(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        offset: i32,
    ) {
        if datasize == 8 {
            self.base.ldursb(64, rt, rn, offset as _);
        } else if datasize == 16 {
            self.base.ldursh(64, rt, rn, offset as _);
        } else {
            self.load_unscaled_immediate(datasize, rt, rn, offset)
        }
    }

    fn load_signed_addressed_by_unsigned_immediate(
        &mut self,
        datasize: i32,
        rt: RegisterId,
        rn: RegisterId,
        pimm: usize,
    ) {
        if datasize == 8 {
            self.base.ldsrb_pimm(64, rt, rn, pimm);
        } else if datasize == 16 {
            self.base.ldrsh_pimm(64, rt, rn, pimm);
        } else {
            self.load_unsigned_immediate(datasize, rt, rn, pimm)
        }
    }

    fn load_link(&mut self, datasize: i32, src: RegisterId, dest: RegisterId) {
        self.base.ldxr(datasize, src, dest);
    }

    fn load_link_acq(&mut self, datasize: i32, src: RegisterId, dest: RegisterId) {
        self.base.ldaxr(datasize, src, dest);
    }

    fn store_cond(&mut self, datasize: i32, src: RegisterId, dest: RegisterId, result: RegisterId) {
        self.base.stxr(datasize, result, src, dest);
    }

    fn store_cond_rel(
        &mut self,
        datasize: i32,
        src: RegisterId,
        dest: RegisterId,
        result: RegisterId,
    ) {
        self.base.stlxr(datasize, result, src, dest);
    }

    pub fn zero_extend(&mut self, datasize: i32, src: GR, dest: GR) {
        if datasize == 8 {
            self.zero_extend_8_to_32(src, dest);
        } else if datasize == 16 {
            self.zero_extend_16_to_32(src, dest)
        } else {
            self.mov(src, dest);
        }
    }

    pub fn zero_extend_8_to_32(&mut self, src: GR, dest: GR) {
        self.base.uxtb(32, dest.0, src.0)
    }

    pub fn zero_extend_16_to_32(&mut self, src: GR, dest: GR) {
        self.base.uxth(32, dest.0, src.0)
    }

    pub fn branch(
        &mut self,
        datasize: i32,
        cond: RelationalCondition,
        left: GR,
        right: GR,
    ) -> Jump {
        self.branch32(cond, left, right)
    }

    pub fn branch32(&mut self, cond: RelationalCondition, left: GR, right: GR) -> Jump {
        self.base.cmp(32, left.0, right.0);
        self.make_branch_rel(cond)
    }

    pub fn branch32_imm(&mut self, cond: RelationalCondition, left: GR, right: i32) -> Jump {
        if right == 0 {
            if let Some(result_condition) = compute_compare_to_zero_into_test(cond) {}
        }

        if is_uint12(right as _) {
            self.base.cmp_imm12(32, left.0, right, 0);
        } else if is_uint12(-right as isize) {
            self.base.cmn_imm12(32, left.0, -right, 0)
        } else {
            let mut r = *self.data_temp_memory_register();
            self.move_to_cached_reg32(right, &mut r);
            self.cached_data_memory_temp_register = r;
            self.base.cmp(32, left.0, r.reg)
        }

        self.make_branch_rel(cond)
    }
    pub fn branch_test32(&mut self, cond: ResultCondition, reg: GR, mask: GR) -> Jump {
        if reg == mask && (cond == ResultCondition::Zero || cond == ResultCondition::NonZero) {
            return self.make_compare_and_branch(32, unsafe { std::mem::transmute(cond) }, reg);
        }

        self.base.tst(32, reg.0, mask.0);
        self.make_branch_res(cond)
    }

    pub fn branch_test32_imm(&mut self, cond: ResultCondition, reg: GR, imm: i32) -> Jump {
        if imm == -1 {
            if cond == ResultCondition::Zero || cond == ResultCondition::NonZero {
                return self.make_compare_and_branch(32, unsafe { std::mem::transmute(cond) }, reg);
            }
            self.base.tst(32, reg.0, reg.0);
        } else if has_one_bit_set!(imm)
            && (cond == ResultCondition::Zero || cond == ResultCondition::NonZero)
        {
            return self.make_test_bit_and_branch(32, reg, imm.trailing_zeros() as _, unsafe {
                std::mem::transmute(cond)
            });
        } else {
            let logical_imm = ARM64LogicalImmediate::create32(imm as _);

            if logical_imm.is_valid() {
                self.base.tst_imm(32, reg.0, logical_imm.value);
                return self.make_branch_res(cond);
            }

            let r = self.get_cached_data_memory_temp_register_and_invalidate();
            self.mov_imm32(imm, r);
            self.base.tst(32, reg.0, r.0);
        }

        self.make_branch_res(cond)
    }

    pub fn sign_extend_32_to_ptr(&mut self, src: GR, dest: GR) {
        self.base.sxtw(32, dest.0, src.0);
    }

    pub fn sign_extend32_to_ptr_imm(&mut self, imm: i32, dest: GR) {
        self.mov_imm64(imm as _, dest);
    }

    pub fn load32(&mut self, addr: Address, dest: GR) {
        if self.try_load_with_offset(32, dest.0, addr.base, addr.offset) {
            return;
        }

        let _ = self.get_cached_memory_temp_register_and_invalidate();

        self.sign_extend32_to_ptr_imm(addr.offset, MEMORY_TEMP_REGISTER);
        self.base.ldr(32, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load32_base(&mut self, addr: BaseIndex, dest: GR) {
        if addr.scale == Scale::One || addr.scale == Scale::Four {
            if let Some(base_gpr) = self.try_fold_base_and_offset_part(addr) {
                self.base.ldr_extended(
                    32,
                    dest.0,
                    base_gpr,
                    addr.index,
                    index_extend_type(addr),
                    addr.scale as _,
                );
                return;
            }
        }
        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base.add_extend(
            64,
            SetFlags::DontSet,
            MEMORY_TEMP_REGISTER.0,
            MEMORY_TEMP_REGISTER.0,
            addr.index,
            index_extend_type(addr),
            addr.scale as _,
        );
        self.base.ldr(32, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load32_addr(&mut self, addr: *const u8, dest: GR) {
        self.load(32, addr as _, dest);
    }

    pub fn load32_pre_index(&mut self, addr: PreIndexAddress, dest: GR) {
        self.base.ldr_pre_index(32, dest.0, addr.base, addr.index);
    }

    pub fn load32_post_index(&mut self, addr: PostIndexAddress, dest: GR) {
        self.base.ldr_post_index(32, dest.0, addr.base, addr.index);
    }

    pub fn load32_with_unaligned_half_words(&mut self, addr: BaseIndex, dest: GR) {
        self.load32_base(addr, dest);
    }

    pub fn load16(&mut self, addr: Address, dest: GR) {
        if self.try_load_with_offset(16, dest.0, addr.base, addr.offset) {
            return;
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset as _, r);
        self.base.ldrh(dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load16_base(&mut self, addr: BaseIndex, dest: GR) {
        if addr.scale == Scale::One || addr.scale == Scale::Four {
            if let Some(base_gpr) = self.try_fold_base_and_offset_part(addr) {
                self.base.ldrh_extend(
                    dest.0,
                    base_gpr,
                    addr.index,
                    index_extend_type(addr),
                    addr.scale as _,
                );
                return;
            }
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base.add_extend(
            64,
            SetFlags::DontSet,
            MEMORY_TEMP_REGISTER.0,
            MEMORY_TEMP_REGISTER.0,
            addr.index,
            index_extend_type(addr),
            addr.scale as _,
        );
        self.base.ldrh(dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load16_extended(&mut self, addr: ExtendedAddress, dest: GR) {
        let mut r = self.cached_memory_temp_register;
        self.move_to_cached_reg64(addr.offset as _, &mut r);
        self.cached_memory_temp_register = r;

        self.base
            .ldrh_extend(dest.0, MEMORY_TEMP_REGISTER.0, addr.base, ExtendOp::UXTX, 1);
        if dest == MEMORY_TEMP_REGISTER {
            let mut r = self.cached_memory_temp_register;
            r.invalidate(self);
            self.cached_memory_temp_register = r;
        }
    }

    pub fn load16_addr(&mut self, addr: *const u8, dest: GR) {
        self.load(16, addr as _, dest);
    }

    pub fn load16_unaligned(&mut self, addr: Address, dest: GR) {
        self.load16(addr, dest);
    }

    pub fn load16_unaligned_base(&mut self, addr: BaseIndex, dest: GR) {
        self.load16_base(addr, dest)
    }

    pub fn load16_sign_extend_to_32(&mut self, addr: Address, dest: GR) {
        if self.try_load_signed_with_offset(16, dest.0, addr.base, addr.offset) {
            return;
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base
            .ldrsh(32, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load16_sign_extend_to_32_base(&mut self, addr: BaseIndex, dest: GR) {
        if addr.scale == Scale::One || addr.scale == Scale::Four {
            if let Some(base_gpr) = self.try_fold_base_and_offset_part(addr) {
                self.base.ldrsh_extended(
                    32,
                    dest.0,
                    base_gpr,
                    addr.index,
                    index_extend_type(addr),
                    addr.scale as _,
                );
                return;
            }
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base.add_extend(
            64,
            SetFlags::DontSet,
            MEMORY_TEMP_REGISTER.0,
            MEMORY_TEMP_REGISTER.0,
            addr.index,
            index_extend_type(addr),
            addr.scale as _,
        );
        self.base
            .ldrsh(32, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load16_sign_extend_to_32_addr(&mut self, addr: *const u8, dest: GR) {
        let mut r = self.cached_memory_temp_register;
        self.move_to_cached_reg64(addr as _, &mut r);
        self.cached_memory_temp_register = r;
        self.base.ldrsh(32, dest.0, MEMORY_TEMP_REGISTER.0, zr);
        if dest == MEMORY_TEMP_REGISTER {
            let mut r = self.cached_memory_temp_register;
            r.invalidate(self);
            self.cached_memory_temp_register = r;
        }
    }

    pub fn sign_extend_16_to_32(&mut self, src: GR, dest: GR) {
        self.base.sxth(32, dest.0, src.0);
    }

    pub fn load8(&mut self, addr: Address, dest: GR) {
        if self.try_load_with_offset(8, dest.0, addr.base, addr.offset) {
            return;
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset as _, r);
        self.base.ldrb(dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load8_base(&mut self, addr: BaseIndex, dest: GR) {
        if addr.scale == Scale::One {
            if let Some(base_gpr) = self.try_fold_base_and_offset_part(addr) {
                self.base.ldrb_extend(
                    dest.0,
                    base_gpr,
                    addr.index,
                    index_extend_type(addr),
                    addr.scale as _,
                );
                return;
            }
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base.add_extend(
            64,
            SetFlags::DontSet,
            MEMORY_TEMP_REGISTER.0,
            MEMORY_TEMP_REGISTER.0,
            addr.index,
            index_extend_type(addr),
            addr.scale as _,
        );
        self.base.ldrb(dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load8_addr(&mut self, addr: *const u8, dest: GR) {
        let mut r = self.cached_memory_temp_register;
        self.move_to_cached_reg64(addr as _, &mut r);
        self.cached_memory_temp_register = r;
        self.base.ldrb(dest.0, MEMORY_TEMP_REGISTER.0, zr);
        if dest == MEMORY_TEMP_REGISTER {
            let mut r = self.cached_memory_temp_register;
            r.invalidate(self);
            self.cached_memory_temp_register = r;
        }
    }

    pub fn load8_post_index(&mut self, src: GR, simm: i32, dest: GR) {
        self.base.ldrb_post_index(dest.0, src.0, simm)
    }

    pub fn load8_signed_extend_to_32(&mut self, addr: Address, dest: GR) {
        if self.try_load_signed_with_offset(8, dest.0, addr.base, addr.offset) {
            return;
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base
            .ldsrb(32, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load8_siogned_extend_to_32_base(&mut self, addr: BaseIndex, dest: GR) {
        if addr.scale == Scale::One || addr.scale == Scale::Four {
            if let Some(base_gpr) = self.try_fold_base_and_offset_part(addr) {
                self.base
                    .ldsrb_extend(32, dest.0, base_gpr, addr.index, index_extend_type(addr));
                return;
            }
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base.add_extend(
            64,
            SetFlags::DontSet,
            MEMORY_TEMP_REGISTER.0,
            MEMORY_TEMP_REGISTER.0,
            addr.index,
            index_extend_type(addr),
            addr.scale as _,
        );
        self.base
            .ldsrb(32, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load8_signed_extend_to_32_addr(&mut self, addr: *const u8, dest: GR) {
        let mut r = self.cached_memory_temp_register;
        self.move_to_cached_reg64(addr as _, &mut r);
        self.cached_memory_temp_register = r;
        self.base.ldsrb(32, dest.0, MEMORY_TEMP_REGISTER.0, zr);
        if dest == MEMORY_TEMP_REGISTER {
            let mut r = self.cached_memory_temp_register;
            r.invalidate(self);
            self.cached_memory_temp_register = r;
        }
    }

    pub fn sign_extend_8_to_32(&mut self, src: GR, dest: GR) {
        self.base.sxtb(32, dest.0, src.0);
    }

    pub fn store64(&mut self, src: GR, addr: Address) {
        if self.try_store_with_offset(64, src.0, addr.base, addr.offset) {
            return;
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base.str(64, src.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn store64_base(&mut self, src: GR, addr: BaseIndex) {
        if addr.scale == Scale::One {
            if let Some(base_gpr) = self.try_fold_base_and_offset_part(addr) {
                self.base.str_extend(
                    64,
                    src.0,
                    base_gpr,
                    addr.index,
                    index_extend_type(addr),
                    addr.scale as _,
                );
                return;
            }
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset, r);
        self.base.add_extend(
            64,
            SetFlags::DontSet,
            MEMORY_TEMP_REGISTER.0,
            MEMORY_TEMP_REGISTER.0,
            addr.index,
            index_extend_type(addr),
            addr.scale as _,
        );
        self.base.str(64, src.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn store64_pre_index(&mut self, src: GR, addr: PreIndexAddress) {
        self.base.str_pre_index(64, src.0, addr.base, addr.index)
    }

    pub fn store64_post_index(&mut self, src: GR, addr: PostIndexAddress) {
        self.base.str_post_index(64, src.0, addr.base, addr.index)
    }

    pub fn store64_addr(&mut self, src: GR, addr: *const u8) {
        self.store(64, src, addr as _);
    }

    pub fn store64_imm_addr(&mut self, src: i64, addr: *const u8) {
        if src == 0 {
            self.store64_addr(GR(zr), addr);
            return;
        }
        let mut r = *self.data_temp_memory_register();
        self.move_to_cached_reg64(src, &mut r);
        self.cached_data_memory_temp_register = r;
        self.store64_addr(DATA_TEMP_REGISTER, addr);
    }

    pub fn store64_imm(&mut self, src: i64, addr: Address) {
        if src == 0 {
            self.store64(GR(zr), addr);
        }
        let mut r = *self.data_temp_memory_register();
        self.move_to_cached_reg64(src, &mut r);
        self.cached_data_memory_temp_register = r;
        self.store64(DATA_TEMP_REGISTER, addr);
    }

    pub fn store64_imm32(&mut self, src: i32, addr: Address) {
        self.store64_imm(src as _, addr);
    }

    pub fn store64_imm_base(&mut self, src: i64, addr: BaseIndex) {
        if src == 0 {
            self.store64_base(GR(zr), addr);
            return;
        }
        let mut r = *self.data_temp_memory_register();
        self.move_to_cached_reg64(src, &mut r);
        self.cached_data_memory_temp_register = r;
        self.store64_base(DATA_TEMP_REGISTER, addr);
    }

    pub fn load64(&mut self, addr: Address, dest: GR) {
        if self.try_load_with_offset(64, dest.0, addr.base, addr.offset) {
            return;
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset as _, r);
        self.base.ldr(64, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load64_base(&mut self, addr: BaseIndex, dest: GR) {
        if addr.scale == Scale::One {
            if let Some(base_gpr) = self.try_fold_base_and_offset_part(addr) {
                self.base.ldr_extended(
                    64,
                    dest.0,
                    base_gpr,
                    addr.index,
                    index_extend_type(addr),
                    addr.scale as _,
                );
                return;
            }
        }

        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend32_to_ptr_imm(addr.offset as _, r);
        self.base.add_extend(
            64,
            SetFlags::DontSet,
            MEMORY_TEMP_REGISTER.0,
            MEMORY_TEMP_REGISTER.0,
            addr.index,
            index_extend_type(addr),
            addr.scale as _,
        );
        self.base.ldr(64, dest.0, addr.base, MEMORY_TEMP_REGISTER.0);
    }

    pub fn load64_addr(&mut self, addr: *const u8, dest: GR) {
        self.load(64, addr as _, dest);
    }

    pub fn load64_pre_index(&mut self, addr: PreIndexAddress, dest: GR) {
        self.base.ldr_pre_index(64, dest.0, addr.base, addr.index)
    }

    pub fn load64_post_index(&mut self, addr: PostIndexAddress, dest: GR) {
        self.base.ldr_post_index(64, dest.0, addr.base, addr.index)
    }

    pub fn load64_with_address_offset_patch(&mut self, addr: Address, dest: GR) -> DataLabel32 {
        let label = DataLabel32::new(self);
        let r = self.get_cached_memory_temp_register_and_invalidate();
        self.sign_extend_32_to_ptr_with_fixed_width(addr.offset, r.0);
        self.base.ldr_extended(
            64,
            dest.0,
            addr.base,
            MEMORY_TEMP_REGISTER.0,
            ExtendOp::SXTW,
            0,
        );
        label
    }

    pub fn load64_with_compact_address_offset_patch(
        &mut self,
        addr: Address,
        dest: GR,
    ) -> DataLabelCompact {
        let label = DataLabelCompact::new(self);
        self.base.ldr_pimm(64, dest.0, addr.base, addr.offset as _);
        label
    }

    pub fn load_pair32_offset(&mut self, src: GR, offset: i32, dest1: GR, dest2: GR) {
        if is_valid_ldpimm(32, offset) {
            self.base.ldp_offset(32, dest1.0, dest2.0, src.0, offset);
            return;
        }
        if src == dest1 {
            self.load32(Address::new(src.0, offset + 4), dest2);
            self.load32(Address::new(src.0, offset), dest1);
        } else {
            self.load32(Address::new(src.0, offset), dest1);
            self.load32(Address::new(src.0, offset + 4), dest2);
        }
    }

    pub fn load_pair32(&mut self, src: GR, dest1: GR, dest2: GR) {
        self.load_pair32_offset(src, 0, dest1, dest2)
    }

    pub fn load_pair64_offset(&mut self, src: GR, offset: i32, dest1: GR, dest2: GR) {
        if is_valid_ldpimm(64, offset) {
            self.base.ldp_offset(64, dest1.0, dest2.0, src.0, offset);
            return;
        }
        if src == dest1 {
            self.load64(Address::new(src.0, offset + 8), dest2);
            self.load64(Address::new(src.0, offset), dest1);
        } else {
            self.load64(Address::new(src.0, offset), dest1);
            self.load64(Address::new(src.0, offset + 8), dest2);
        }
    }

    pub fn load_pair64(&mut self, src: GR, dest1: GR, dest2: GR) {
        self.load_pair64_offset(src, 0, dest1, dest2)
    }
}

fn compute_compare_to_zero_into_test(cond: RelationalCondition) -> Option<ResultCondition> {
    match cond {
        RelationalCondition::Equal => Some(ResultCondition::Zero),
        RelationalCondition::NotEqual => Some(ResultCondition::NonZero),
        RelationalCondition::LessThan => Some(ResultCondition::Signed),
        RelationalCondition::GreaterThanOrEqual => Some(ResultCondition::PositiveOrZero),
        _ => None,
    }
}

fn index_extend_type(addr: BaseIndex) -> ExtendOp {
    match addr.extend {
        Extend::ZExt32 => ExtendOp::UXTW,
        Extend::SExt32 => ExtendOp::SXTW,
        Extend::None => ExtendOp::UXTX,
    }
}

#[allow(non_upper_case_globals)]
pub const t0: GR = GR(x0);
#[allow(non_upper_case_globals)]
pub const t1: GR = GR(x1);
#[allow(non_upper_case_globals)]
pub const t2: GR = GR(x2);
#[allow(non_upper_case_globals)]
pub const t3: GR = GR(x3);
#[allow(non_upper_case_globals)]
pub const t4: GR = GR(x4);
#[allow(non_upper_case_globals)]
pub const t5: GR = GR(x5);
#[allow(non_upper_case_globals)]
pub const t6: GR = GR(x6);
#[allow(non_upper_case_globals)]
pub const t7: GR = GR(x7);
#[allow(non_upper_case_globals)]
pub const t8: GR = GR(x8);
#[allow(non_upper_case_globals)]
pub const t9: GR = GR(x9);
#[allow(non_upper_case_globals)]
pub const t10: GR = GR(x10);
#[allow(non_upper_case_globals)]
pub const t11: GR = GR(x11);
#[allow(non_upper_case_globals)]
pub const t12: GR = GR(x12);
#[allow(non_upper_case_globals)]
pub const t13: GR = GR(x13);
#[allow(non_upper_case_globals)]
pub const t14: GR = GR(x14);
#[allow(non_upper_case_globals)]
pub const t15: GR = GR(x16);

#[allow(non_upper_case_globals)]
pub const a0: GR = GR(x0);
#[allow(non_upper_case_globals)]
pub const a1: GR = GR(x1);
#[allow(non_upper_case_globals)]
pub const a2: GR = GR(x2);
#[allow(non_upper_case_globals)]
pub const a3: GR = GR(x3);
#[allow(non_upper_case_globals)]
pub const a4: GR = GR(x4);
#[allow(non_upper_case_globals)]
pub const a5: GR = GR(x5);
#[allow(non_upper_case_globals)]
pub const a6: GR = GR(x6);
#[allow(non_upper_case_globals)]
pub const a7: GR = GR(x7);

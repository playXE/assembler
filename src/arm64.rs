#![allow(non_snake_case)]
use std::{mem::size_of, mem::transmute};

pub const fn max_jump_replacement_size() -> usize {
    4
}

use crate::{
    assembler_buffer::{AssemblerBuffer, AssemblerLabel},
    assembler_common::{is_int9, is_uint12, is_valid_signed_imm7},
    macro_assembler::{Assembler, Jump, Label},
};

#[macro_use]
pub mod registers;

pub type RegisterId = i8;

macro_rules! register_id {
    (($id: ident, $name: expr, $r: expr, $cs: expr), $($rest:tt)*) => {
        #[allow(non_upper_case_globals)]
        pub const $id: RegisterId = 0;
        register_id!(@parse 1; $($rest)*);
    };
    (@parse $i: expr; ($id: ident, $name: expr, $r: expr, $cs: expr), $($rest:tt)*) => {
        #[allow(non_upper_case_globals)]
        pub const $id: RegisterId = $i;
        register_id!(@parse $i + 1; $($rest)*);
    };
    (@parse $i: expr;) => {

    }
}

for_each_gp_register!(register_id);

macro_rules! alias_register_id {
    ($(($name: ident, $e: expr, $alias: expr)),+) => {
        $(
            #[allow(non_upper_case_globals)]
            pub const $name: RegisterId = $alias;
        )*
    };

}

for_each_register_alias!(alias_register_id);

macro_rules! sp_register_id {
    (($id: ident, $name: expr), $($rest:tt)*) => {
        #[allow(non_upper_case_globals)]
        pub const $id: RegisterId = 0;
        sp_register_id!(@parse 1; $($rest)*);
    };
    (@parse $i: expr; ($id: ident, $name: expr), $($rest:tt)*) => {
        #[allow(non_upper_case_globals)]
        pub const $id: RegisterId = $i;
        sp_register_id!(@parse $i + 1; $($rest)*);
    };
    (@parse $i: expr;) => {

    }
}

for_each_sp_register!(sp_register_id);

for_each_fp_register!(register_id);

pub const INVALID_FP_REG: i8 = -1;
pub const INVALID_GP_REG: i8 = -1;
pub const INVALID_SP_REG: i8 = -1;

pub fn is_valid_ldpimm(datasize: i32, immediate: i32) -> bool {
    let shift = mem_pair_offset_shifted(false, mem_pair_op_size_int(datasize));
    is_valid_signed_imm7(immediate, shift as _)
}

pub fn is_valid_fpimm(datasize: i32, immediate: i32) -> bool {
    let shift = mem_pair_offset_shifted(false, mem_pair_op_size_fp(datasize));
    is_valid_signed_imm7(immediate, shift as _)
}

pub const fn is_sp(reg: RegisterId) -> bool {
    reg == sp
}

pub const fn is_zr(reg: RegisterId) -> bool {
    reg == zr
}

pub const fn is_4byte_aligned(addr: usize) -> bool {
    addr & 0x3 == 0
}

pub const fn is_uint5(imm: i32) -> bool {
    (imm & !0x1f) == 0
}

pub struct UInt5 {
    value: i32,
}

impl UInt5 {
    pub fn new(value: i32) -> Self {
        value.into()
    }
}

impl From<UInt5> for i32 {
    fn from(x: UInt5) -> Self {
        x.value
    }
}

impl Into<UInt5> for i32 {
    fn into(self) -> UInt5 {
        assert!(is_uint5(self));
        UInt5 { value: self }
    }
}

pub struct UInt12 {
    value: i32,
}

impl UInt12 {
    pub fn new(value: i32) -> Self {
        value.into()
    }
}

impl From<UInt12> for i32 {
    fn from(x: UInt12) -> Self {
        x.value
    }
}

impl Into<UInt12> for i32 {
    fn into(self) -> UInt12 {
        assert!(is_uint12(self as _));
        UInt12 { value: self }
    }
}

pub struct PostIndex {
    value: i32,
}

impl PostIndex {
    pub fn new(value: i32) -> Self {
        value.into()
    }
}

impl From<PostIndex> for i32 {
    fn from(x: PostIndex) -> Self {
        x.value
    }
}

impl Into<PostIndex> for i32 {
    fn into(self) -> PostIndex {
        assert!(is_int9(self));
        PostIndex { value: self }
    }
}

pub struct PreIndex {
    value: i32,
}

impl PreIndex {
    pub fn new(value: i32) -> Self {
        value.into()
    }
}

impl From<PreIndex> for i32 {
    fn from(x: PreIndex) -> Self {
        x.value
    }
}

impl Into<PreIndex> for i32 {
    fn into(self) -> PreIndex {
        assert!(is_int9(self));
        PreIndex { value: self }
    }
}

pub struct PairPreIndex {
    value: i32,
}

impl PairPreIndex {
    pub fn new(value: i32) -> Self {
        value.into()
    }
}

impl From<PairPreIndex> for i32 {
    fn from(x: PairPreIndex) -> Self {
        x.value
    }
}

impl Into<PairPreIndex> for i32 {
    fn into(self) -> PairPreIndex {
        assert!(is_int!(11, self));
        PairPreIndex { value: self }
    }
}

pub fn get_half_word(value: u64, which: i32) -> u16 {
    (value.wrapping_shr((which << 4) as u32)) as _
}

pub fn get_half_word32(value: u32, which: i32) -> u16 {
    (value >> (which << 4) as u32) as _
}

const fn jump_enum_with_size(index: usize, value: usize) -> usize {
    (value << 4) | index
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum JumpType {
    Fixed = jump_enum_with_size(0, 0),
    NoCondition = jump_enum_with_size(1, size_of::<u32>()),
    Condition = jump_enum_with_size(2, 2 * size_of::<u32>()),
    CompareAndBranch = jump_enum_with_size(3, size_of::<u32>() * 2),
    TestBit = jump_enum_with_size(4, size_of::<u32>() * 2),
    NoConditionFixedSize = jump_enum_with_size(5, size_of::<u32>() * 1),
    ConditionFixedSize = jump_enum_with_size(6, size_of::<u32>() * 2),
    CompareAndBranchFixedSize = jump_enum_with_size(7, size_of::<u32>() * 2),
    TestBitFixedSize = jump_enum_with_size(8, size_of::<u32>() * 2),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum JumpLinkType {
    Invalid = jump_enum_with_size(0, 0),
    NoCondition = jump_enum_with_size(1, 1 * size_of::<u32>()),
    ConditionDirect = jump_enum_with_size(2, 1 * size_of::<u32>()),
    Condition = jump_enum_with_size(3, 2 * size_of::<u32>()),
    CompareAndBranch = jump_enum_with_size(4, 2 * size_of::<u32>()),
    CompareAndBranchDirect = jump_enum_with_size(5, 1 * size_of::<u32>()),
    TestBit = jump_enum_with_size(6, 2 * size_of::<u32>()),
    TestBitDirect = jump_enum_with_size(7, 1 * size_of::<u32>()),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Condition {
    EQ,
    NE,
    HS,
    LO,
    MI,
    PL,
    VS,
    VC,
    HI,
    LS,
    GE,
    LT,
    GT,
    LE,
    AL,
    Invalid,
}

pub const CS: Condition = Condition::HS;
pub const CC: Condition = Condition::LO;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub struct LinkRecord {
    pub from: i64,
    pub to: i64,
    pub compare_register: RegisterId,
    pub typ: JumpType,
    pub link: JumpLinkType,
    pub condition: Condition,
    pub bitnumber: usize,
    pub is64bit: bool,
}

impl LinkRecord {
    pub fn condition(from: isize, to: isize, typ: JumpType, condition: Condition) -> Self {
        Self {
            from: from as _,
            to: to as _,
            compare_register: INVALID_GP_REG,
            typ,
            link: JumpLinkType::Invalid,
            condition,
            bitnumber: 0,
            is64bit: false,
        }
    }

    pub fn condition_and_compare(
        from: isize,
        to: isize,
        typ: JumpType,
        condition: Condition,
        is64bit: bool,
        compare_register: RegisterId,
    ) -> Self {
        Self {
            from: from as _,
            to: to as _,
            compare_register,
            typ,
            link: JumpLinkType::Invalid,
            condition,
            bitnumber: 0,
            is64bit,
        }
    }

    pub fn test_bit(
        from: isize,
        to: isize,
        typ: JumpType,
        condition: Condition,
        bitnumber: usize,
        compare_register: RegisterId,
    ) -> Self {
        Self {
            from: from as _,
            to: to as _,
            compare_register,
            typ,
            link: JumpLinkType::Invalid,
            condition,
            bitnumber,
            is64bit: false,
        }
    }
}

/// bits(N) VFPExpandImm(bits(8) imm8);
//
/// Encoding of floating point immediates is a litte complicated. Here's a
/// high level description:
///     +/-m*2-n where m and n are integers, 16 <= m <= 31, 0 <= n <= 7
/// and the algirithm for expanding to a single precision float:
///     return imm8<7>:NOT(imm8<6>):Replicate(imm8<6>,5):imm8<5:0>:Zeros(19);
//
/// The trickiest bit is how the exponent is handled. The following table
/// may help clarify things a little:
///     654
///     100 01111100 124 -3 1020 01111111100
///     101 01111101 125 -2 1021 01111111101
///     110 01111110 126 -1 1022 01111111110
///     111 01111111 127  0 1023 01111111111
///     000 10000000 128  1 1024 10000000000
///     001 10000001 129  2 1025 10000000001
///     010 10000010 130  3 1026 10000000010
///     011 10000011 131  4 1027 10000000011
/// The first column shows the bit pattern stored in bits 6-4 of the arm
/// encoded immediate. The second column shows the 8-bit IEEE 754 single
/// -precision exponent in binary, the third column shows the raw decimal
/// value. IEEE 754 single-precision numbers are stored with a bias of 127
/// to the exponent, so the fourth column shows the resulting exponent.
/// From this was can see that the exponent can be in the range -3..4,
/// which agrees with the high level description given above. The fifth
/// and sixth columns shows the value stored in a IEEE 754 double-precision
/// number to represent these exponents in decimal and binary, given the
/// bias of 1023.
//
/// Ultimately, detecting doubles that can be encoded as immediates on arm
/// and encoding doubles is actually not too bad. A floating point value can
/// be encoded by retaining the sign bit, the low three bits of the exponent
/// and the high 4 bits of the mantissa. To validly be able to encode an
/// immediate the remainder of the mantissa must be zero, and the high part
/// of the exponent must match the top bit retained, bar the highest bit
/// which must be its inverse.
pub fn can_encode_fp_imm(d: f64) -> bool {
    let masked = d.to_bits() & 0x7fc0ffffffffffff;
    masked == 0x3fc0000000000000 || masked == 0x4000000000000000
}

pub fn encode_fp_imm(d: f64) -> i32 {
    let u = d.to_bits();
    ((u >> 56) as i32 & 0x80) | ((u >> 48) as i32 & 0x7f)
}

pub struct ARM64Assembler {
    buffer: AssemblerBuffer,
    index_of_last_watchpoint: usize,
    index_of_tail_of_last_watchpoint: usize,
    jumps_to_link: Vec<LinkRecord>,
}

impl ARM64Assembler {
    pub fn buffer(&self) -> &AssemblerBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut AssemblerBuffer {
        &mut self.buffer
    }
    pub fn new() -> Self {
        Self {
            buffer: AssemblerBuffer::new(),
            index_of_last_watchpoint: 0,
            index_of_tail_of_last_watchpoint: 0,
            jumps_to_link: vec![],
        }
    }
    pub fn link_jump_condition(
        &mut self,
        from: AssemblerLabel,
        to: AssemblerLabel,
        typ: JumpType,
        condition: Condition,
    ) {
        self.jumps_to_link.push(LinkRecord::condition(
            from.offset() as _,
            to.offset() as _,
            typ,
            condition,
        ))
    }

    pub fn link_jump_compare(
        &mut self,
        from: AssemblerLabel,
        to: AssemblerLabel,
        typ: JumpType,
        condition: Condition,
        is64bit: bool,
        compare_register: RegisterId,
    ) {
        self.jumps_to_link.push(LinkRecord::condition_and_compare(
            from.offset() as _,
            to.offset() as _,
            typ,
            condition,
            is64bit,
            compare_register,
        ))
    }

    pub fn link_jump_test_bit(
        &mut self,
        from: AssemblerLabel,
        to: AssemblerLabel,
        typ: JumpType,
        condition: Condition,
        bitnumber: usize,
        compare_register: RegisterId,
    ) {
        self.jumps_to_link.push(LinkRecord::test_bit(
            from.offset() as _,
            to.offset() as _,
            typ,
            condition,
            bitnumber,
            compare_register,
        ))
    }

    pub fn link_jump_(&mut self, j: crate::macro_assembler::Jump) {
        if j.typ == JumpType::CompareAndBranch || j.typ == JumpType::CompareAndBranchFixedSize {
            let to = self.label();
            self.link_jump_compare(j.label, to, j.typ, j.cond, j.is64bit, j.compare_register);
        } else if j.typ == JumpType::TestBit || j.typ == JumpType::TestBitFixedSize {
            let to = self.label();
            self.link_jump_test_bit(j.label, to, j.typ, j.cond, j.bitnumber, j.compare_register);
        } else {
            let to = self.label();
            self.link_jump_condition(j.label, to, j.typ, j.cond);
        }
    }

    pub fn link_to(&mut self, jump: Jump, label: Label) {
        if jump.typ == JumpType::CompareAndBranch || jump.typ == JumpType::CompareAndBranchFixedSize
        {
            self.link_jump_compare(
                jump.label,
                label.label,
                jump.typ,
                jump.cond,
                jump.is64bit,
                jump.compare_register,
            );
        } else if jump.typ == JumpType::TestBit || jump.typ == JumpType::TestBitFixedSize {
            self.link_jump_test_bit(
                jump.label,
                label.label,
                jump.typ,
                jump.cond,
                jump.bitnumber,
                jump.compare_register,
            );
        } else {
            self.link_jump_condition(jump.label, label.label, jump.typ, jump.cond);
        }
    }

    pub fn label(&mut self) -> AssemblerLabel {
        let mut result = self.buffer.label();
        while result.offset() < self.index_of_tail_of_last_watchpoint as u32 {
            self.nop();
            result = self.buffer.label();
        }

        result
    }
    pub fn label_ignoring_watchpoints(&mut self) -> AssemblerLabel {
        self.buffer.label()
    }

    pub fn label_for_watchpoint(&mut self) -> AssemblerLabel {
        let mut result = self.buffer.label();
        if result.offset() as usize != self.index_of_last_watchpoint {
            result = self.label();
        }
        self.index_of_last_watchpoint = result.offset() as _;
        self.index_of_tail_of_last_watchpoint =
            result.offset() as usize + max_jump_replacement_size();
        result
    }
    pub fn align(&mut self, alignment: usize) -> AssemblerLabel {
        while !self.buffer.is_aligned(alignment) {
            self.brk(0);
        }

        self.label()
    }

    pub fn get_relocated_address(code: *mut u8, label: AssemblerLabel) -> *mut u8 {
        unsafe { code.add(label.offset() as _) }
    }

    pub fn get_difference_between_labels(a: AssemblerLabel, b: AssemblerLabel) -> isize {
        b.offset() as isize - a.offset() as isize
    }
    pub fn get_call_return_offset(call: AssemblerLabel) -> u32 {
        call.offset()
    }
    pub fn insn(&mut self, instruction: i32) {
        self.buffer.put_int(instruction);
    }

    pub fn nop(&mut self) {
        self.insn(nop_pseudo())
    }

    pub fn adc(
        &mut self,
        sf: i32,
        set_flags: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
    ) {
        self.insn(add_subtract_with_carry(
            datasize(sf),
            AddOp::ADD,
            set_flags,
            rm,
            rn,
            rd,
        ));
    }

    pub fn add_imm12(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        imm12: i32,
        shift: i32,
    ) {
        self.insn(add_subtract_immediate(
            datasize(sf),
            AddOp::ADD,
            s,
            shift,
            imm12,
            rn,
            rd,
        ))
    }

    pub fn add(&mut self, sf: i32, s: SetFlags, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.add_shifted(sf, s, rd, rn, rm, ShiftType::LSL, 0);
    }
    pub fn add_extend(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(add_subtract_extended_register(
            datasize(sf),
            AddOp::ADD,
            s,
            rm,
            extend,
            amount,
            rn,
            rd,
        ))
    }

    pub fn add_shifted(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        if is_sp(rd) || is_sp(rn) {
            self.add_extend(sf, s, rd, rn, rm, ExtendOp::UXTX, amount);
        } else {
            self.insn(add_subtract_shifted_register(
                datasize(sf),
                AddOp::ADD,
                s,
                shift,
                rm,
                amount,
                rn,
                rd,
            ));
        }
    }

    pub fn sub_imm12(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        imm12: i32,
        shift: i32,
    ) {
        self.insn(add_subtract_immediate(
            datasize(sf),
            AddOp::SUB,
            s,
            shift,
            imm12,
            rn,
            rd,
        ))
    }

    pub fn sub_shifted(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.insn(add_subtract_shifted_register(
            datasize(sf),
            AddOp::SUB,
            s,
            shift,
            rm,
            amount,
            rn,
            rd,
        ))
    }

    pub fn sub_extend(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(add_subtract_extended_register(
            datasize(sf),
            AddOp::SUB,
            s,
            rm,
            extend,
            amount,
            rn,
            rd,
        ))
    }

    pub fn sub(&mut self, sf: i32, s: SetFlags, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        if is_sp(rd) || is_sp(rn) {
            self.sub_extend(sf, s, rd, rn, rm, ExtendOp::UXTX, 0);
        } else {
            self.sub_shifted(sf, s, rd, rn, rm, ShiftType::LSL, 0);
        }
    }

    pub fn adr(&mut self, rd: RegisterId, offset: i32) {
        self.insn(pc_relative(false, offset, rd));
    }

    pub fn adrp(&mut self, rd: RegisterId, offset: i32) {
        self.insn(pc_relative(true, offset >> 12, rd));
    }

    pub fn and_shifted(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.insn(logical_shifted_register(
            datasize(sf),
            if SetFlags::S == s {
                LogicalOp::ANDS
            } else {
                LogicalOp::AND
            },
            shift,
            false,
            rm,
            amount,
            rn,
            rd,
        ));
    }

    pub fn and_immediate(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        imm: i32,
    ) {
        self.insn(logical_immediate(
            datasize(sf),
            if SetFlags::S == s {
                LogicalOp::ANDS
            } else {
                LogicalOp::AND
            },
            imm,
            rn,
            rd,
        ));
    }

    pub fn and(&mut self, sf: i32, s: SetFlags, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.and_shifted(sf, s, rd, rn, rm, ShiftType::LSL, 0);
    }

    pub fn asrv(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(data_processing_2source(
            datasize(sf),
            rm,
            DataOp2Source::ASRV,
            rn,
            rd,
        ))
    }

    pub fn asr(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.asrv(sf, rd, rn, rm);
    }

    pub fn b(&mut self) {
        self.insn(unconditional_branch_immediate(false, 0));
    }

    pub fn b_cond(&mut self, cond: Condition, offset: i32) {
        self.insn(conditional_branch_immediate(offset, cond));
    }

    pub fn bfm(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, immr: i32, imms: i32) {
        self.insn(bitfield(datasize(sf), BitfieldOp::BFM, immr, imms, rn, rd));
    }

    pub fn bfi(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, lsb: i32, width: i32) {
        self.bfm(sf, rd, rn, (sf as i32 - lsb) & (sf as i32 - 1), width - 1);
    }

    pub fn bfc(&mut self, sf: i32, rd: RegisterId, lsb: i32, width: i32) {
        self.bfi(sf, rd, zr, lsb, width);
    }

    pub fn bfxil(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, lsb: i32, width: i32) {
        self.bfm(sf, rd, rn, lsb, lsb + width - 1);
    }

    pub fn bic_shifted(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.insn(logical_shifted_register(
            datasize(sf),
            if s == SetFlags::S {
                LogicalOp::ANDS
            } else {
                LogicalOp::AND
            },
            shift,
            true,
            rm,
            amount,
            rn,
            rd,
        ))
    }

    pub fn bic(&mut self, sf: i32, s: SetFlags, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.bic_shifted(sf, s, rd, rn, rm, ShiftType::LSL, 0);
    }

    pub fn bl(&mut self) {
        self.insn(unconditional_branch_immediate(true, 0));
    }

    pub fn blr(&mut self, rn: RegisterId) {
        self.insn(unconditional_branch_register(BranchType::CALL, rn));
    }

    pub fn br(&mut self, rn: RegisterId) {
        self.insn(unconditional_branch_register(BranchType::JMP, rn));
    }

    pub fn brk(&mut self, imm: u16) {
        self.insn(excepn_generation(ExcepnOp::BREAKPOINT, imm, 0));
    }

    pub unsafe fn is_brk(address: *mut u8) -> bool {
        let expected = excepn_generation(ExcepnOp::BREAKPOINT, 0, 0);
        let mask = excepn_generation_imm_mask();
        let candidate = address.cast::<i32>().read();
        (candidate & mask) == expected
    }

    pub fn cbnz(&mut self, sf: i32, rt: RegisterId, mut offset: i32) {
        offset >>= 2;
        self.insn(compare_and_branch_immediate(datasize(sf), true, offset, rt));
    }

    pub fn cbz(&mut self, sf: i32, rt: RegisterId, mut offset: i32) {
        offset >>= 2;
        self.insn(compare_and_branch_immediate(
            datasize(sf),
            false,
            offset,
            rt,
        ));
    }

    pub fn ccmn(&mut self, sf: i32, rn: RegisterId, rm: RegisterId, vnzcv: i32, cond: Condition) {
        self.insn(conditional_compare_register(
            datasize(sf),
            AddOp::ADD,
            rm,
            cond,
            rn,
            vnzcv,
        ));
    }

    pub fn ccmn_imm(&mut self, sf: i32, rn: RegisterId, imm: i32, vnzcv: i32, cond: Condition) {
        self.insn(conditional_compare_immediate(
            datasize(sf),
            AddOp::ADD,
            imm,
            cond,
            rn,
            vnzcv,
        ));
    }

    pub fn ccmp(&mut self, sf: i32, rn: RegisterId, rm: RegisterId, vnzcv: i32, cond: Condition) {
        self.insn(conditional_compare_register(
            datasize(sf),
            AddOp::SUB,
            rm,
            cond,
            rn,
            vnzcv,
        ));
    }

    pub fn ccmp_imm(&mut self, sf: i32, rn: RegisterId, imm: i32, vnzcv: i32, cond: Condition) {
        self.insn(conditional_compare_immediate(
            datasize(sf),
            AddOp::SUB,
            imm,
            cond,
            rn,
            vnzcv,
        ));
    }

    pub fn csinc(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        cond: Condition,
    ) {
        self.insn(conditional_select(
            datasize(sf),
            false,
            rm,
            cond,
            true,
            rn,
            rd,
        ));
    }

    pub fn csinv(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        cond: Condition,
    ) {
        self.insn(conditional_select(
            datasize(sf),
            true,
            rm,
            cond,
            false,
            rn,
            rd,
        ));
    }

    pub fn csneg(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        cond: Condition,
    ) {
        self.insn(conditional_select(
            datasize(sf),
            true,
            rm,
            cond,
            true,
            rn,
            rd,
        ));
    }

    pub fn cinc(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        cond: Condition,
    ) {
        self.csinc(sf, rd, rn, rm, invert(cond))
    }

    pub fn cinv(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        cond: Condition,
    ) {
        self.csinv(sf, rd, rn, rm, invert(cond))
    }

    pub fn cls(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.insn(data_processing_1source(
            datasize(sf),
            DataOp1Source::CLS,
            rn,
            rd,
        ))
    }

    pub fn clz(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.insn(data_processing_1source(
            datasize(sf),
            DataOp1Source::CLZ,
            rn,
            rd,
        ))
    }

    pub fn cmn_imm12(&mut self, sf: i32, rn: RegisterId, imm12: i32, shift: i32) {
        self.add_imm12(sf, SetFlags::S, zr, rn, imm12, shift)
    }

    pub fn cmn(&mut self, sf: i32, rn: RegisterId, rm: RegisterId) {
        self.add(sf, SetFlags::S, zr, rn, rm)
    }

    pub fn cmn_extended(
        &mut self,
        sf: i32,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.add_extend(sf, SetFlags::S, zr, rn, rm, extend, amount)
    }

    pub fn cmp_imm12(&mut self, sf: i32, rn: RegisterId, imm12: i32, shift: i32) {
        self.sub_imm12(sf, SetFlags::S, zr, rn, imm12, shift)
    }

    pub fn cmp(&mut self, sf: i32, rn: RegisterId, rm: RegisterId) {
        self.sub(sf, SetFlags::S, zr, rn, rm)
    }

    pub fn cmp_extended(
        &mut self,
        sf: i32,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.sub_extend(sf, SetFlags::S, zr, rn, rm, extend, amount)
    }

    pub fn cneg(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, cond: Condition) {
        self.csneg(sf, rd, rn, rn, invert(cond))
    }

    pub fn csel(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        cond: Condition,
    ) {
        self.insn(conditional_select(
            datasize(sf),
            false,
            rm,
            cond,
            false,
            rn,
            rd,
        ))
    }

    pub fn cset(&mut self, sf: i32, rd: RegisterId, cond: Condition) {
        self.csinc(sf, rd, zr, zr, invert(cond));
    }

    pub fn csetm(&mut self, sf: i32, rd: RegisterId, cond: Condition) {
        self.csinv(sf, rd, zr, zr, invert(cond));
    }

    pub fn eon_shifted(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.insn(logical_shifted_register(
            datasize(sf),
            LogicalOp::EOR,
            shift,
            true,
            rm,
            amount,
            rn,
            rd,
        ))
    }

    pub fn eon(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.eon_shifted(sf, rd, rn, rm, ShiftType::LSL, 0)
    }

    pub fn eor_shifted(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.insn(logical_shifted_register(
            datasize(sf),
            LogicalOp::EOR,
            shift,
            false,
            rm,
            amount,
            rn,
            rd,
        ))
    }

    pub fn eor(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.eor_shifted(sf, rd, rn, rm, ShiftType::LSL, 0)
    }

    pub fn eor_imm(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(logical_immediate(datasize(sf), LogicalOp::EOR, imm, rn, rd))
    }

    pub fn extr(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId, lsb: i32) {
        self.insn(extract(datasize(sf), rm, lsb, rn, rd))
    }

    pub fn hint(&mut self, hint: i32) {
        self.insn(hint_pseudo(hint))
    }

    pub fn hlt(&mut self, imm: u16) {
        self.insn(excepn_generation(ExcepnOp::HALT as _, imm, 0))
    }

    pub fn illegal_instruction(&mut self) {
        self.insn(0);
    }

    pub fn ldp_post_index(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        simm: i32,
    ) {
        self.insn(load_store_register_pair_post_index(
            mem_pair_op_size_int(sf),
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn ldp_pre_indx(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        simm: i32,
    ) {
        self.insn(load_store_register_pair_pre_index(
            mem_pair_op_size_int(sf),
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn ldp_offset(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        offset: i32,
    ) {
        self.insn(load_store_register_pair_offset(
            mem_pair_op_size_int(sf),
            false,
            MemOp::LOAD,
            offset,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn ldnp(&mut self, sf: i32, rt: RegisterId, rt2: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pair_non_temporal(
            mem_pair_op_size_int(sf),
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn ldr(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.ldr_extended(sf, rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn ldr_extended(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset(
            memopsize(sf),
            false,
            MemOp::LOAD,
            rm,
            extend,
            amount != 0,
            rn,
            rt,
        ))
    }

    pub fn ldr_pimm(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            memopsize(sf),
            false,
            MemOp::LOAD,
            encode_positive_immediate(sf, pimm),
            rn,
            rt,
        ))
    }

    pub fn ldr_post_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index(
            memopsize(sf),
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ))
    }

    pub fn ldr_pre_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index(
            memopsize(sf),
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ))
    }
    pub fn ldr_literal(&mut self, sf: i32, rt: RegisterId, offset: i32) {
        self.insn(load_register_literal(
            if sf == 64 {
                LDR_LITERAL_OP_64BIT
            } else {
                LDR_LITERAL_OP_32BIT
            },
            false,
            offset >> 2,
            rt,
        ))
    }

    pub fn ldrb(&mut self, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(load_store_register_offset(
            MemOpSIZE::S8or128,
            false,
            MemOp::LOAD,
            rm,
            ExtendOp::UXTX,
            false,
            rn,
            rt,
        ));
    }

    pub fn ldrb_extend(
        &mut self,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        let _ = amount;
        self.insn(load_store_register_offset(
            MemOpSIZE::S8or128,
            false,
            MemOp::LOAD,
            rm,
            extend,
            false,
            rn,
            rt,
        ));
    }

    pub fn ldrb_pimm(&mut self, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            MemOpSIZE::S8or128,
            false,
            MemOp::LOAD,
            encode_positive_immediate(8, pimm),
            rn,
            rt,
        ));
    }

    pub fn ldrb_post_index(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index(
            MemOpSIZE::S8or128,
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrb_pre_index(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index(
            MemOpSIZE::S8or128,
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrh(&mut self, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.ldrh_extend(rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn ldrh_extend(
        &mut self,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset(
            MemOpSIZE::S16,
            false,
            MemOp::LOAD,
            rm,
            extend,
            amount == 1,
            rn,
            rt,
        ));
    }

    pub fn ldrh_pimm(&mut self, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            MemOpSIZE::S16,
            false,
            MemOp::LOAD,
            encode_positive_immediate(16, pimm),
            rn,
            rt,
        ));
    }

    pub fn ldrh_post_index(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index(
            MemOpSIZE::S16,
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrh_pre_index(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index(
            MemOpSIZE::S16,
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldsrb(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(load_store_register_offset(
            MemOpSIZE::S8or128,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            rm,
            ExtendOp::UXTX,
            false,
            rn,
            rt,
        ));
    }

    pub fn ldsrb_extend(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
    ) {
        self.insn(load_store_register_offset(
            MemOpSIZE::S8or128,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            rm,
            extend,
            true,
            rn,
            rt,
        ));
    }

    pub fn ldsrb_pimm(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            MemOpSIZE::S8or128,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            encode_positive_immediate(8, pimm),
            rn,
            rt,
        ));
    }

    pub fn ldsrb_post_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index(
            MemOpSIZE::S8or128,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldsrb_pre_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index(
            MemOpSIZE::S8or128,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrsh_extended(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset(
            MemOpSIZE::S16,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            rm,
            extend,
            amount == 1,
            rn,
            rt,
        ));
    }

    pub fn ldrsh(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.ldrsh_extended(sf, rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn ldrsh_pimm(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            MemOpSIZE::S16,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            encode_positive_immediate(16, pimm),
            rn,
            rt,
        ));
    }

    pub fn ldrsh_post_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index(
            MemOpSIZE::S16,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrsh_pre_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index(
            MemOpSIZE::S16,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrsw_extended(
        &mut self,

        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset(
            MemOpSIZE::S32,
            false,
            LOAD_SIGNED64,
            rm,
            extend,
            amount == 2,
            rn,
            rt,
        ));
    }

    pub fn ldrsw(&mut self, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.ldrsw_extended(rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn ldrsw_pimm(&mut self, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            MemOpSIZE::S32,
            false,
            LOAD_SIGNED64,
            encode_positive_immediate(32, pimm),
            rn,
            rt,
        ));
    }

    pub fn ldrsw_post_index(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index(
            MemOpSIZE::S32,
            false,
            LOAD_SIGNED64,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrsw_pre_index(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index(
            MemOpSIZE::S32,
            false,
            LOAD_SIGNED64,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldrsw_literal(&mut self, rt: RegisterId, offset: i32) {
        self.insn(load_register_literal(
            LDR_LITERAL_OP_LDRSW,
            false,
            offset >> 2,
            rt,
        ));
    }

    pub fn ldur(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            memopsize(sf),
            true,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldurb(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            MemOpSIZE::S8or128,
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldurh(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            MemOpSIZE::S16,
            false,
            MemOp::LOAD,
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldursb(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            MemOpSIZE::S8or128,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldursh(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            MemOpSIZE::S16,
            false,
            if sf == 64 {
                LOAD_SIGNED64
            } else {
                LOAD_SIGNED32
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldursw(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            MemOpSIZE::S32,
            false,
            LOAD_SIGNED64,
            simm,
            rn,
            rt,
        ));
    }

    pub fn lsl_imm(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, shift: i32) {
        let datasize = sf;
        self.ubfm(
            sf,
            rd,
            rn,
            (datasize - shift) & (datasize - 1),
            datasize - 1 - shift,
        )
    }

    pub fn lsl(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.lslv(sf, rd, rn, rm);
    }

    pub fn lslv(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(data_processing_2source(
            datasize(sf),
            rm,
            DataOp2Source::LSLV,
            rn,
            rd,
        ));
    }

    pub fn lsr_imm(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, shift: i32) {
        self.ubfm(sf, rd, rn, shift, sf - 1)
    }
    pub fn lsr(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.lsrv(sf, rd, rn, rm);
    }
    pub fn lsrv(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(data_processing_2source(
            datasize(sf),
            rm,
            DataOp2Source::LSRV,
            rn,
            rd,
        ));
    }

    pub fn madd(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        ra: RegisterId,
    ) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::MADD,
            rm,
            ra,
            rn,
            rd,
        ));
    }
    pub fn msub(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        ra: RegisterId,
    ) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::MSUB,
            rm,
            ra,
            rn,
            rd,
        ));
    }

    pub fn mneg(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.msub(sf, rd, rn, rm, zr);
    }

    pub fn mov(&mut self, sf: i32, rd: RegisterId, rm: RegisterId) {
        if is_sp(rd) || is_sp(rm) {
            self.add_imm12(sf, SetFlags::DontSet, rd, rm, 0, 0);
        } else {
            self.orr(sf, rd, zr, rm);
        }
    }

    pub fn movi(&mut self, sf: i32, rd: RegisterId, imm: i32) {
        self.orr_imm(sf, rd, zr, imm)
    }

    pub fn movk(&mut self, sf: i32, rd: RegisterId, imm: u16, imm_shift: i32) {
        self.insn(move_wide_immediate(
            datasize(sf),
            MoveWideOp::K,
            imm_shift >> 4,
            imm,
            rd,
        ));
    }

    pub fn movn(&mut self, sf: i32, rd: RegisterId, imm: u16, imm_shift: i32) {
        self.insn(move_wide_immediate(
            datasize(sf),
            MoveWideOp::N,
            imm_shift >> 4,
            imm,
            rd,
        ));
    }

    pub fn movz(&mut self, sf: i32, rd: RegisterId, imm: u16, imm_shift: i32) {
        self.insn(move_wide_immediate(
            datasize(sf),
            MoveWideOp::Z,
            imm_shift >> 4,
            imm,
            rd,
        ));
    }

    pub fn mul(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.madd(sf, rd, rn, rm, zr);
    }

    pub fn mvn(&mut self, sf: i32, rd: RegisterId, rm: RegisterId) {
        self.orn(sf, rd, zr, rm)
    }

    pub fn mvn_shifted(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.orn_shifted(sf, rd, zr, rm, shift, amount)
    }
    pub fn neg(&mut self, sf: i32, s: SetFlags, rd: RegisterId, rm: RegisterId) {
        self.sub(sf, s, rd, zr, rm);
    }

    pub fn neg_shifted(
        &mut self,
        sf: i32,
        s: SetFlags,
        rd: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.sub_shifted(sf, s, rd, zr, rm, shift, amount)
    }

    pub fn ngc(&mut self, sf: i32, s: SetFlags, rd: RegisterId, rm: RegisterId) {
        self.sbc(sf, s, rd, zr, rm);
    }

    pub fn orn(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.orn_shifted(sf, rd, rn, rm, ShiftType::LSL, 0);
    }

    pub fn orn_shifted(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.insn(logical_shifted_register(
            datasize(sf),
            LogicalOp::ORR,
            shift,
            true,
            rm,
            amount,
            rn,
            rd,
        ))
    }

    pub fn orr_shifted(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.insn(logical_shifted_register(
            datasize(sf),
            LogicalOp::ORR,
            shift,
            false,
            rm,
            amount,
            rn,
            rd,
        ))
    }

    pub fn orr(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.orr_shifted(sf, rd, rn, rm, ShiftType::LSL, 0);
    }

    pub fn orr_imm(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(logical_immediate(datasize(sf), LogicalOp::ORR, imm, rn, rd))
    }

    pub fn ubfm(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, immr: i32, imms: i32) {
        self.insn(bitfield(datasize(sf), BitfieldOp::UBFM, immr, imms, rn, rd));
    }

    pub fn rbit(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.insn(data_processing_1source(
            datasize(sf),
            DataOp1Source::RBIT,
            rn,
            rd,
        ))
    }

    pub fn ret(&mut self, rn: RegisterId) {
        self.insn(unconditional_branch_register(BranchType::RET, rn));
    }

    pub fn rev(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        if sf == 32 {
            self.insn(data_processing_1source(
                Datasize::D32,
                DataOp1Source::REV32,
                rn,
                rd,
            ))
        } else {
            self.insn(data_processing_1source(
                Datasize::D64,
                DataOp1Source::REV64,
                rn,
                rd,
            ))
        }
    }

    pub fn rev16(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.insn(data_processing_1source(
            datasize(sf),
            DataOp1Source::REV16,
            rn,
            rd,
        ))
    }

    pub fn rev32(&mut self, rd: RegisterId, rn: RegisterId) {
        self.insn(data_processing_1source(
            Datasize::D64,
            DataOp1Source::REV32,
            rn,
            rd,
        ))
    }

    pub fn ror(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.rorv(sf, rd, rn, rm);
    }

    pub fn ror_shifted(&mut self, sf: i32, rd: RegisterId, rs: RegisterId, shift: i32) {
        self.extr(sf, rd, rs, rs, shift);
    }

    pub fn rorv(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(data_processing_2source(
            datasize(sf),
            rm,
            DataOp2Source::RORV,
            rn,
            rd,
        ))
    }

    pub fn sbc(&mut self, sf: i32, s: SetFlags, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(add_subtract_with_carry(
            datasize(sf),
            AddOp::SUB,
            s,
            rm,
            rn,
            rd,
        ))
    }

    pub unsafe fn fill_nops(base: *mut u8, size: usize) {
        let mut n = size / size_of::<u32>();

        let mut ptr = base.cast::<u32>();
        while n > 0 {
            let insn = nop_pseudo();
            ptr.write(insn as _);
            ptr = ptr.add(1);
            n -= 1;
        }
    }

    pub fn dmbish(&mut self) {
        self.insn(0xd5033bbfu32 as _);
    }

    pub fn dmbishst(&mut self) {
        self.insn(0xd5033abfu32 as _);
    }

    pub fn ldar(&mut self, sf: i32, src: RegisterId, dst: RegisterId) {
        self.insn(exotic_load(
            memopsize(sf),
            ExoticLoadFence::Acquire,
            ExoticLoadAtomic::None,
            dst,
            src,
        ))
    }

    pub fn ldxr(&mut self, sf: i32, src: RegisterId, dst: RegisterId) {
        self.insn(exotic_load(
            memopsize(sf),
            ExoticLoadFence::None,
            ExoticLoadAtomic::Link,
            dst,
            src,
        ))
    }

    pub fn ldaxr(&mut self, sf: i32, src: RegisterId, dst: RegisterId) {
        self.insn(exotic_load(
            memopsize(sf),
            ExoticLoadFence::Acquire,
            ExoticLoadAtomic::Link,
            dst,
            src,
        ))
    }

    pub fn stxr(&mut self, sf: i32, result: RegisterId, src: RegisterId, dst: RegisterId) {
        self.insn(exotic_store(
            memopsize(sf),
            ExoticStoreFence::None,
            result,
            src,
            dst,
        ))
    }
    pub fn stlxr(&mut self, sf: i32, result: RegisterId, src: RegisterId, dst: RegisterId) {
        self.insn(exotic_store(
            memopsize(sf),
            ExoticStoreFence::Release,
            result,
            src,
            dst,
        ))
    }
    pub fn stlr(&mut self, sf: i32, src: RegisterId, dst: RegisterId) {
        self.insn(store_release(memopsize(sf), src, dst))
    }

    pub fn sbfm(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, immr: i32, imms: i32) {
        self.insn(bitfield(datasize(sf), BitfieldOp::SBFM, immr, imms, rn, rd))
    }

    pub fn sbfiz(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, lsb: i32, width: i32) {
        let datasize = sf;
        self.sbfm(sf, rd, rn, (datasize - lsb) & (datasize - 1), width - 1);
    }

    pub fn sbfx(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, lsb: i32, width: i32) {
        self.sbfm(sf, rd, rn, lsb, lsb + width - 1);
    }

    pub fn sdiv(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(data_processing_2source(
            datasize(sf),
            rm,
            DataOp2Source::SDIV,
            rn,
            rd,
        ))
    }

    pub fn smaddl(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        ra: RegisterId,
    ) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::SMADDL,
            rm,
            ra,
            rn,
            rd,
        ))
    }

    pub fn smnegl(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.smsubl(sf, rd, rn, rm, zr);
    }

    pub fn smsubl(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        ra: RegisterId,
    ) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::SMSUBL,
            rm,
            ra,
            rn,
            rd,
        ))
    }

    pub fn smulh(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        ra: RegisterId,
    ) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::SMULH,
            rm,
            ra,
            rn,
            rd,
        ))
    }
    pub fn stp_pair_post_index(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        imm: i32,
    ) {
        self.insn(load_store_register_pair_post_index(
            mem_pair_op_size_int(sf),
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn stp_pair_pre_index(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        imm: i32,
    ) {
        self.insn(load_store_register_pair_pre_index(
            mem_pair_op_size_int(sf),
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn stp_offset(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        imm: i32,
    ) {
        self.insn(load_store_register_pair_offset(
            mem_pair_op_size_int(sf),
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn stnp(&mut self, sf: i32, rt: RegisterId, rt2: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_pair_non_temporal(
            mem_pair_op_size_int(sf),
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn stp_fp_post_index(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        imm: i32,
    ) {
        self.insn(load_store_register_pair_post_index(
            mem_pair_op_size_fp(sf),
            true,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn stp_fp_pre_index(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        imm: i32,
    ) {
        self.insn(load_store_register_pair_pre_index(
            mem_pair_op_size_fp(sf),
            true,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn stp_fp_offset(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rt2: RegisterId,
        rn: RegisterId,
        imm: i32,
    ) {
        self.insn(load_store_register_pair_offset(
            mem_pair_op_size_fp(sf),
            true,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }

    pub fn stnp_fp(&mut self, sf: i32, rt: RegisterId, rt2: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_pair_non_temporal(
            mem_pair_op_size_fp(sf),
            true,
            MemOp::STORE,
            imm,
            rn,
            rt,
            rt2,
        ))
    }
    pub fn str_extend(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset_r(
            memopsize(sf),
            false,
            MemOp::STORE,
            rm,
            extend,
            amount != 0,
            rn,
            rt,
        ))
    }

    pub fn str(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.str_extend(sf, rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn str_pimm(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, imm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            memopsize(sf),
            false,
            MemOp::STORE,
            encode_positive_immediate(sf, imm) as _,
            rn,
            rt,
        ))
    }

    pub fn str_post_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_post_index(
            memopsize(sf),
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
        ))
    }

    pub fn str_pre_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_pre_index(
            memopsize(sf),
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
        ))
    }

    pub fn strb(&mut self, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(load_store_register_offset(
            MemOpSIZE::S8or128,
            false,
            MemOp::STORE,
            rm,
            ExtendOp::UXTX,
            false,
            rn,
            rt,
        ))
    }

    pub fn strb_extended(
        &mut self,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        let _ = amount;
        self.insn(load_store_register_offset_r(
            MemOpSIZE::S8or128,
            false,
            MemOp::STORE,
            rm,
            extend,
            false,
            rn,
            rt,
        ))
    }

    pub fn strb_pimm(&mut self, rt: RegisterId, rn: RegisterId, imm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            MemOpSIZE::S8or128,
            false,
            MemOp::STORE,
            encode_positive_immediate(8, imm) as _,
            rn,
            rt,
        ))
    }

    pub fn strb_post_index(&mut self, rt: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_post_index(
            MemOpSIZE::S8or128,
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
        ))
    }

    pub fn strb_pre_index(&mut self, rt: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_pre_index(
            MemOpSIZE::S8or128,
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
        ))
    }

    pub fn strh(&mut self, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.strh_extended(rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn strh_extended(
        &mut self,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset_r(
            MemOpSIZE::S16,
            false,
            MemOp::STORE,
            rm,
            extend,
            amount == 1,
            rn,
            rt,
        ))
    }

    pub fn strh_pimm(&mut self, rt: RegisterId, rn: RegisterId, imm: usize) {
        self.insn(load_store_register_unsigned_immediate(
            MemOpSIZE::S16,
            false,
            MemOp::STORE,
            encode_positive_immediate(16, imm) as _,
            rn,
            rt,
        ))
    }

    pub fn strh_post_index(&mut self, rt: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_post_index(
            MemOpSIZE::S16,
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
        ))
    }

    pub fn strh_pre_index(&mut self, rt: RegisterId, rn: RegisterId, imm: i32) {
        self.insn(load_store_register_pre_index(
            MemOpSIZE::S16,
            false,
            MemOp::STORE,
            imm,
            rn,
            rt,
        ))
    }

    pub fn stur(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            memopsize(sf),
            false,
            MemOp::STORE,
            simm as _,
            rn,
            rt,
        ))
    }

    pub fn sturb(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            MemOpSIZE::S8or128,
            false,
            MemOp::STORE,
            simm as _,
            rn,
            rt,
        ))
    }

    pub fn sturb_imm(&mut self, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate(
            MemOpSIZE::S16,
            false,
            MemOp::STORE,
            simm as _,
            rn,
            rt,
        ))
    }

    pub fn sxtb(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.sbfm(sf, rd, rn, 0, 7);
    }

    pub fn sxth(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.sbfm(sf, rd, rn, 0, 15);
    }

    pub fn sxtw(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.sbfm(sf, rd, rn, 0, 31);
    }

    pub fn tbz(&mut self, rt: RegisterId, imm: i32, mut offset: i32) {
        offset >>= 2;
        self.insn(test_and_branch_immediate(false, imm, offset, rt))
    }

    pub fn tbnz(&mut self, rt: RegisterId, imm: i32, mut offset: i32) {
        offset >>= 2;
        self.insn(test_and_branch_immediate(true, imm, offset, rt))
    }

    pub fn tst(&mut self, sf: i32, rn: RegisterId, rm: RegisterId) {
        self.and(sf, SetFlags::S, zr, rn, rm);
    }

    pub fn tst_shifted(
        &mut self,
        sf: i32,
        rn: RegisterId,
        rm: RegisterId,
        shift: ShiftType,
        amount: i32,
    ) {
        self.and_shifted(sf, SetFlags::S, zr, rn, rm, shift, amount);
    }

    pub fn tst_imm(&mut self, sf: i32, rn: RegisterId, imm: i32) {
        self.and_immediate(sf, SetFlags::S, zr, rn, imm)
    }

    pub fn ubfiz(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, lsb: i32, width: i32) {
        self.ubfm(sf, rd, rn, (sf - lsb) & (sf - 1), width - 1)
    }
    pub fn ubfx(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, lsb: i32, width: i32) {
        self.ubfm(sf, rd, rn, lsb, lsb + width - 1)
    }

    pub fn udiv(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(data_processing_2source(
            datasize(sf),
            rm,
            DataOp2Source::UDIV,
            rn,
            rd,
        ));
    }

    pub fn umaddl(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        ra: RegisterId,
    ) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::UMADDL,
            rm,
            ra,
            rn,
            rd,
        ));
    }

    pub fn umsubl(
        &mut self,
        sf: i32,
        rd: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        ra: RegisterId,
    ) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::UMSUBL,
            rm,
            ra,
            rn,
            rd,
        ));
    }

    pub fn umnegl(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.umsubl(sf, rd, rn, rm, zr);
    }

    pub fn umulh(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.insn(data_processing_3source(
            datasize(sf),
            DataOp3Source::UMULH,
            rm,
            zr,
            rn,
            rd,
        ));
    }

    pub fn umull(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.umaddl(sf, rd, rn, rm, zr);
    }

    pub fn uxtb(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.ubfm(sf, rd, rn, 0, 7);
    }
    pub fn uxth(&mut self, sf: i32, rd: RegisterId, rn: RegisterId) {
        self.ubfm(sf, rd, rn, 0, 15);
    }

    pub fn uxtw(&mut self, rd: RegisterId, rn: RegisterId) {
        self.ubfm(64, rd, rn, 0, 31);
    }
    pub fn smull(&mut self, sf: i32, rd: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.smaddl(sf, rd, rn, rm, zr);
    }

    // floating point instructions:

    pub fn fabs(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FABS,
            vn,
            vd,
        ));
    }

    pub fn fadd(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FADD,
            vn,
            vd,
        ));
    }

    pub fn fccmp(&mut self, sf: i32, vn: RegisterId, vm: RegisterId, vnzcv: i32, cond: Condition) {
        self.insn(floating_point_conditional_compare(
            datasize(sf),
            vm,
            cond,
            vn,
            FPCondCmpOp::FCMP,
            vnzcv,
        ));
    }

    pub fn fccmpe(&mut self, sf: i32, vn: RegisterId, vm: RegisterId, vnzcv: i32, cond: Condition) {
        self.insn(floating_point_conditional_compare(
            datasize(sf),
            vm,
            cond,
            vn,
            FPCondCmpOp::FCMPE,
            vnzcv,
        ));
    }

    pub fn fcmp(&mut self, sf: i32, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_compare(datasize(sf), vm, vn, FPCmpOp::FCMP));
    }

    pub fn fcmp_0(&mut self, sf: i32, vn: RegisterId) {
        self.insn(floating_point_compare(datasize(sf), 0, vn, FPCmpOp::FCMP0));
    }

    pub fn fcmpe(&mut self, sf: i32, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_compare(datasize(sf), vm, vn, FPCmpOp::FCMPE));
    }

    pub fn fcmpe_0(&mut self, sf: i32, vn: RegisterId) {
        self.insn(floating_point_compare(datasize(sf), 0, vn, FPCmpOp::FCMPE0));
    }

    pub fn fcsel(
        &mut self,
        sf: i32,
        vd: RegisterId,
        vn: RegisterId,
        vm: RegisterId,
        cond: Condition,
    ) {
        self.insn(floating_point_conditional_select(
            datasize(sf),
            vm,
            cond,
            vn,
            vd,
        ));
    }

    pub fn fcvt(&mut self, dstsize: i32, srcsize: i32, vd: RegisterId, vn: RegisterId) {
        let typ = if srcsize == 64 {
            Datasize::D64
        } else if srcsize == 32 {
            Datasize::D32
        } else {
            Datasize::D16
        };
        let opcode = if dstsize == 64 {
            FPDataOp1Source::FCVT2DOUBLE
        } else if dstsize == 32 {
            FPDataOp1Source::FCVT2SINGLE
        } else {
            FPDataOp1Source::FCVT2HALF
        };

        self.insn(floating_point_data_processing_1source(typ, opcode, vn, vd));
    }

    pub fn fcvtas(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTAS,
            vn,
            rd,
        ));
    }

    pub fn fcvtau(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTAU,
            vn,
            rd,
        ));
    }

    pub fn fcvtms(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTMS,
            vn,
            rd,
        ));
    }

    pub fn fcvtmu(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTMU,
            vn,
            rd,
        ));
    }

    pub fn fcvtns(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTNS,
            vn,
            rd,
        ));
    }

    pub fn fcvtnu(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTNU,
            vn,
            rd,
        ));
    }

    pub fn fcvtps(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTPS,
            vn,
            rd,
        ));
    }

    pub fn fcvtpu(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTPU,
            vn,
            rd,
        ));
    }

    pub fn fcvtzs(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTZS,
            vn,
            rd,
        ));
    }

    pub fn fcvtzu(&mut self, dstsize: i32, srcsize: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions(
            datasize(dstsize),
            datasize(srcsize),
            FPIntConvOp::FCVTZU,
            vn,
            rd,
        ));
    }

    pub fn fdiv(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FDIV,
            vn,
            vd,
        ));
    }

    pub fn fmadd(
        &mut self,
        sf: i32,
        vd: RegisterId,
        vn: RegisterId,
        vm: RegisterId,
        va: RegisterId,
    ) {
        self.insn(floating_point_data_processing_3source(
            datasize(sf),
            false,
            vm,
            AddOp::ADD,
            va,
            vn,
            vd,
        ));
    }

    pub fn fmax(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FMAX,
            vn,
            vd,
        ));
    }

    pub fn fmaxnm(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FMAXNM,
            vn,
            vd,
        ));
    }

    pub fn fmin(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FMIN,
            vn,
            vd,
        ));
    }

    pub fn fminnm(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FMINNM,
            vn,
            vd,
        ));
    }

    pub fn fmov(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FMOV,
            vn,
            vd,
        ))
    }

    pub fn fmov_x2q(&mut self, sf: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions_fr(
            datasize(sf),
            datasize(sf),
            FPIntConvOp::FMOVQtoX,
            vn,
            rd,
        ))
    }

    pub fn fmov_imm(&mut self, sf: i32, rd: RegisterId, imm: f64) {
        self.insn(floating_point_immediate(
            datasize(sf),
            encode_fp_imm(imm),
            rd,
        ))
    }

    pub fn fmov_q2x(&mut self, sf: i32, vd: RegisterId, rn: RegisterId) {
        self.insn(floating_point_integer_convertions_rr(
            datasize(sf),
            datasize(sf),
            FPIntConvOp::FMOVXtoQ,
            rn,
            vd,
        ))
    }

    pub fn fmov_x2q_top(&mut self, sf: i32, rd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_integer_convertions_fr(
            datasize(sf),
            datasize(sf),
            FPIntConvOp::FMOVXtoQtop,
            vn,
            rd,
        ))
    }

    pub fn fmov_q2x_top(&mut self, sf: i32, vd: RegisterId, rn: RegisterId) {
        self.insn(floating_point_integer_convertions_rr(
            datasize(sf),
            datasize(sf),
            FPIntConvOp::FMOVQtoXtop,
            rn,
            vd,
        ))
    }

    pub fn fmsub(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_3source(
            datasize(sf),
            false,
            vm,
            AddOp::SUB,
            vn,
            vd,
            vd,
        ));
    }

    pub fn fmul(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FMUL,
            vn,
            vd,
        ));
    }

    pub fn fneg(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FNEG,
            vn,
            vd,
        ));
    }

    pub fn fnmadd(
        &mut self,
        sf: i32,
        vd: RegisterId,
        vn: RegisterId,
        vm: RegisterId,
        va: RegisterId,
    ) {
        self.insn(floating_point_data_processing_3source(
            datasize(sf),
            true,
            vm,
            AddOp::ADD,
            va,
            vn,
            vd,
        ));
    }

    pub fn fnmsub(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_3source(
            datasize(sf),
            true,
            vm,
            AddOp::SUB,
            vn,
            vd,
            vd,
        ));
    }

    pub fn fnmul(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FNMUL,
            vn,
            vd,
        ));
    }

    pub fn vand(&mut self, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(vector_processing_logical(SIMD3SameLogical::AND, vm, vn, vd));
    }

    pub fn vorr(&mut self, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(vector_processing_logical(SIMD3SameLogical::ORR, vm, vn, vd));
    }

    pub fn frinta(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FRINTA,
            vn,
            vd,
        ));
    }

    pub fn frinti(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FRINTI,
            vn,
            vd,
        ));
    }

    pub fn frintm(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FRINTM,
            vn,
            vd,
        ));
    }

    pub fn frintn(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FRINTN,
            vn,
            vd,
        ));
    }

    pub fn frintp(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FRINTP,
            vn,
            vd,
        ));
    }

    pub fn frintx(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FRINTX,
            vn,
            vd,
        ));
    }

    pub fn frintz(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FRINTZ,
            vn,
            vd,
        ));
    }

    pub fn fsqrt(&mut self, sf: i32, vd: RegisterId, vn: RegisterId) {
        self.insn(floating_point_data_processing_1source(
            datasize(sf),
            FPDataOp1Source::FSQRT,
            vn,
            vd,
        ));
    }

    pub fn fsub(&mut self, sf: i32, vd: RegisterId, vn: RegisterId, vm: RegisterId) {
        self.insn(floating_point_data_processing_2source(
            datasize(sf),
            vm,
            FPDataOp2Source::FSUB,
            vn,
            vd,
        ));
    }

    pub fn ldr_f_extended(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::LOADV128
            } else {
                MemOp::LOAD
            },
            rm,
            extend,
            amount != 0,
            rn,
            rt,
        ));
    }

    pub fn ldr_f(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.ldr_f_extended(sf, rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn ldr_f_pimm(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::LOADV128
            } else {
                MemOp::LOAD
            },
            encode_positive_immediate(sf, pimm),
            rn,
            rt,
        ))
    }

    pub fn ldr_f_post_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::LOADV128
            } else {
                MemOp::LOAD
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldr_f_pre_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::LOADV128
            } else {
                MemOp::LOAD
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn ldr_f_literal(&mut self, sf: i32, rt: RegisterId, offset: i32) {
        self.insn(load_register_literal_r(
            if sf == 128 {
                LDR_LITERAL_OP_128BIT
            } else {
                LDR_LITERAL_OP_64BIT
            },
            true,
            offset >> 2,
            rt,
        ))
    }

    pub fn ldur_f(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::LOADV128
            } else {
                MemOp::LOAD
            },
            simm,
            rn,
            rt,
        ))
    }

    pub fn str_f_extended(
        &mut self,
        sf: i32,
        rt: RegisterId,
        rn: RegisterId,
        rm: RegisterId,
        extend: ExtendOp,
        amount: i32,
    ) {
        self.insn(load_store_register_offset_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::STOREV128
            } else {
                MemOp::STORE
            },
            rm,
            extend,
            amount != 0,
            rn,
            rt,
        ));
    }

    pub fn str_f(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, rm: RegisterId) {
        self.str_f_extended(sf, rt, rn, rm, ExtendOp::UXTX, 0);
    }

    pub fn str_f_pimm(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, pimm: usize) {
        self.insn(load_store_register_unsigned_immediate_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::STOREV128
            } else {
                MemOp::STORE
            },
            encode_positive_immediate(sf, pimm),
            rn,
            rt,
        ))
    }

    pub fn str_f_post_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_post_index_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::STOREV128
            } else {
                MemOp::STORE
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn str_f_pre_index(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_pre_index_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::STOREV128
            } else {
                MemOp::STORE
            },
            simm,
            rn,
            rt,
        ));
    }

    pub fn stur_f(&mut self, sf: i32, rt: RegisterId, rn: RegisterId, simm: i32) {
        self.insn(load_store_register_unscaled_immediate_r(
            memopsize(sf),
            true,
            if sf == 128 {
                MemOp::STOREV128
            } else {
                MemOp::STORE
            },
            simm,
            rn,
            rt,
        ))
    }

    pub fn scvtf(&mut self, dstsize: i32, srcsize: i32, vd: RegisterId, rn: RegisterId) {
        self.insn(floating_point_integer_convertions_fr(
            datasize(srcsize),
            datasize(dstsize),
            FPIntConvOp::SCVTF,
            rn,
            vd,
        ))
    }

    pub fn ucvtf(&mut self, dstsize: i32, srcsize: i32, vd: RegisterId, rn: RegisterId) {
        self.insn(floating_point_integer_convertions_fr(
            datasize(srcsize),
            datasize(dstsize),
            FPIntConvOp::UCVTF,
            rn,
            vd,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchType {
    JMP,
    CALL,
    RET,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchTargetType {
    Direct,
    Indirect,
}

pub use BranchType::*;
pub use MoveWideOp::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Datasize {
    D32,
    D64,
    D64Top,
    D16,
    D128,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MemOpSIZE {
    S8or128,
    S16,
    S32,
    S64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AddOp {
    ADD,
    SUB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BitfieldOp {
    SBFM,
    BFM,
    UBFM,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataOp1Source {
    RBIT,
    REV16,
    REV32,
    REV64,
    CLZ,
    CLS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataOp2Source {
    UDIV = 2,
    SDIV = 3,
    LSLV = 8,
    LSRV = 9,
    ASRV = 10,
    RORV = 11,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ShiftType {
    LSL,
    LSR,
    ASR,
    ROR,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataOp3Source {
    MADD = 0,
    MSUB = 1,
    SMADDL = 2,
    SMSUBL = 3,
    SMULH = 4,
    UMADDL = 10,
    UMSUBL = 11,
    UMULH = 12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExcepnOp {
    EXCEPTION = 0,
    BREAKPOINT = 1,
    HALT = 2,
    DCPS = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FPCmpOp {
    FCMP = 0x00,
    FCMP0 = 0x08,
    FCMPE = 0x10,
    FCMPE0 = 0x18,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FPCondCmpOp {
    FCMP,
    FCMPE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FPDataOp1Source {
    FMOV = 0,
    FABS = 1,
    FNEG = 2,
    FSQRT = 3,
    FCVT2SINGLE = 4,
    FCVT2DOUBLE = 5,
    FCVT2HALF = 7,
    FRINTN = 8,
    FRINTP = 9,
    FRINTM = 10,
    FRINTZ = 11,
    FRINTA = 12,
    FRINTX = 14,
    FRINTI = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FPDataOp2Source {
    FMUL,
    FDIV,
    FADD,
    FSUB,
    FMAX,
    FMIN,
    FMAXNM,
    FMINNM,
    FNMUL,
}

pub const SIMD_LOGICAL_OP: u8 = 0x03;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SIMD3SameLogical {
    AND = 0x00,
    BIC = 0x01,
    ORR = 0x02,
    ORN = 0x03,
    EOR = 0x80,
    BSL = 0x81,
    BIT = 0x82,
    BIF = 0x83,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FPIntConvOp {
    FCVTNS = 0x00,
    FCVTNU = 0x01,
    SCVTF = 0x02,
    UCVTF = 0x03,
    FCVTAS = 0x04,
    FCVTAU = 0x05,
    FMOVQtoX = 0x06,
    FMOVXtoQ = 0x07,
    FCVTPS = 0x08,
    FCVTPU = 0x09,
    FMOVQtoXtop = 0x0e,
    FMOVXtoQtop = 0x0f,
    FCVTMS = 0x10,
    FCVTMU = 0x11,
    FCVTZS = 0x18,
    FCVTZU = 0x19,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    AND,
    ORR,
    EOR,
    ANDS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MemOp {
    STORE = 0,
    LOAD = 1,
    STOREV128 = 2,
    LOADV128 = 3,
}

pub const PREFETCH: MemOp = MemOp::STOREV128;
pub const LOAD_SIGNED64: MemOp = MemOp::STOREV128;
pub const LOAD_SIGNED32: MemOp = MemOp::LOADV128;

pub const MEMPAIROP_32: u8 = 0;
pub const MEMPAIROP_LOADSIGNED_32: u8 = 1;
pub const MEMPAIROP_64: u8 = 2;
pub const MEMPAIROP_V32: u8 = MEMPAIROP_32;
pub const MEMPAIROP_V64: u8 = 1;
pub const MEMPAIROP_V128: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MoveWideOp {
    N = 0,
    Z = 2,
    K = 3,
}

pub const LDR_LITERAL_OP_32BIT: u8 = 0;
pub const LDR_LITERAL_OP_64BIT: u8 = 1;
pub const LDR_LITERAL_OP_128BIT: u8 = 2;
pub const LDR_LITERAL_OP_LDRSW: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExoticLoadFence {
    None,
    Acquire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExoticLoadAtomic {
    Link,
    None,
}

pub enum ExoticStoreFence {
    None,
    Release,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExtendOp {
    UXTB,
    UXTH,
    UXTW,
    UXTX,
    SXTB,
    SXTH,
    SXTW,
    SXTX,
}

pub fn datasize(i: i32) -> Datasize {
    if i == 64 {
        Datasize::D64
    } else {
        Datasize::D32
    }
}

pub fn memopsize(i: i32) -> MemOpSIZE {
    match i {
        8 | 128 => MemOpSIZE::S8or128,
        16 => MemOpSIZE::S16,
        32 => MemOpSIZE::S32,
        _ => MemOpSIZE::S64,
    }
}

pub fn mem_pair_offset_shifted(v: bool, size: u8) -> usize {
    if v {
        size as usize + 2
    } else {
        (size as usize >> 1) + 2
    }
}

pub fn mem_pair_op_size_int(sf: i32) -> u8 {
    if sf == 64 {
        MEMPAIROP_64
    } else {
        MEMPAIROP_32
    }
}

pub fn mem_pair_op_size_fp(sf: i32) -> u8 {
    if sf == 128 {
        return MEMPAIROP_V128;
    }
    if sf == 32 {
        MEMPAIROP_V64
    } else {
        MEMPAIROP_V32
    }
}

pub unsafe fn disassemble_nop(addr: *mut u8) -> bool {
    addr.cast::<u32>().read() == 0xd503201f
}

unsafe fn relink_jump_or_call(
    typ: BranchType,
    from: *mut u32,
    from_instruction: *mut u32,
    to: *mut u8,
) {
    if typ == BranchType::JMP && disassemble_nop(from.cast()) {
        let mut op01 = false;
        let mut imm19 = 0;
        let mut condition = Condition::AL;
        let is_conditional_branch_immediate = disassemble_conditional_branch_immediate(
            from.sub(1).cast(),
            &mut op01,
            &mut imm19,
            &mut condition,
        );
        if is_conditional_branch_immediate {
            if imm19 == 8 {
                condition = invert(condition);
            }

            link_conditional_branch(
                BranchTargetType::Indirect,
                condition,
                from.sub(1),
                from_instruction.sub(1),
                to,
            );
            return;
        }

        let mut opsize = Datasize::D16;
        let mut op = false;
        let mut rt = 0;
        let is_compare_and_branch_immediate = disassemble_compare_and_branch_immediate(
            from.sub(1),
            &mut opsize,
            &mut op,
            &mut imm19,
            &mut rt,
        );

        if is_compare_and_branch_immediate {
            if imm19 == 8 {
                op = !op;
            }

            link_compare_and_branch(
                BranchTargetType::Indirect,
                if op { Condition::NE } else { Condition::EQ },
                opsize == Datasize::D64,
                rt,
                from.sub(1),
                from_instruction.sub(1),
                to,
            );
            return;
        }

        let mut imm14 = 0;
        let mut bitnumber = 0;
        let is_test_and_branch_immediate = disassemble_test_and_branch_immediate(
            from.sub(1),
            &mut op,
            &mut bitnumber,
            &mut imm14,
            &mut rt,
        );

        if is_test_and_branch_immediate {
            if imm14 == 8 {
                op = !op;
            }

            link_test_and_branch(
                BranchTargetType::Indirect,
                if op { Condition::NE } else { Condition::EQ },
                bitnumber,
                rt,
                from.sub(1),
                from_instruction.sub(1),
                to,
            );
            return;
        }
    }
    link_jump_or_call(typ, from, from_instruction, to)
}

pub unsafe fn address_of(code: *mut u8, label: AssemblerLabel) -> *mut u32 {
    code.add(label.offset() as _).cast()
}
pub fn disassemble_xor_sp(reg: i32) -> RegisterId {
    if reg == 31 {
        sp
    } else {
        reg as _
    }
}

pub fn disassemble_xor_zr(reg: i32) -> RegisterId {
    if reg == 31 {
        zr
    } else {
        reg as _
    }
}

pub fn disassemble_xor_zr_or_sp(use_zr: bool, reg: i32) -> RegisterId {
    if reg == 31 {
        if use_zr {
            zr
        } else {
            sp
        }
    } else {
        reg as _
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SetFlags {
    DontSet,
    S,
}

pub unsafe fn disassemble_add_subtract_immediate(
    address: *mut u32,
    sf: &mut Datasize,
    op: &mut AddOp,
    s: &mut SetFlags,
    shift: &mut i32,
    imm12: &mut i32,
    rn: &mut RegisterId,
    rd: &mut RegisterId,
) -> bool {
    let instruction = address.cast::<i32>().read();
    *sf = transmute((instruction >> 31) & 1);
    *op = transmute((instruction >> 30) & 1);
    *s = transmute((instruction >> 29) & 1);

    *shift = (instruction >> 22) & 3;
    *imm12 = (instruction >> 10) & 0x3ff;

    *rn = disassemble_xor_sp((instruction >> 5) & 0x1f);
    *rd = disassemble_xor_zr_or_sp(*s as i32 != 0, instruction & 0x1f);
    (instruction & 0x1f000000) == 0x11000000
}

pub unsafe fn disassemble_load_store_register_unsigned_immediate(
    address: *mut u32,
    size: &mut MemOpSIZE,
    v: &mut bool,
    opc: &mut MemOp,
    imm12: &mut i32,
    rn: &mut RegisterId,
    rt: &mut RegisterId,
) -> bool {
    let insn = address.cast::<i32>().read();
    *size = transmute((insn >> 30) & 3);
    *v = ((insn >> 26) & 1) != 0;
    *opc = transmute((insn >> 22) & 3);
    *imm12 = (insn >> 10) & 0xfff;

    *rn = disassemble_xor_sp((insn >> 5) & 0x1f);
    *rt = disassemble_xor_zr(insn & 0x1f);
    (insn & 0x3b000000) == 0x39000000
}

pub unsafe fn disassemble_mov_with_imediate(
    address: *mut u32,
    sf: &mut Datasize,
    opc: &mut MoveWideOp,
    hw: &mut i32,
    imm16: &mut u16,
    rd: &mut RegisterId,
) -> bool {
    let insn = address.cast::<i32>().read();
    *sf = transmute((insn >> 31) & 1);
    *opc = transmute((insn >> 29) & 3);
    *hw = (insn >> 21) & 0x3;
    *imm16 = (insn >> 5) as _;
    *rd = disassemble_xor_zr(insn & 0x1f);
    (insn & 0x1f800000) == 0x12800000
}

pub unsafe fn disassemble_compare_and_branch_immediate(
    address: *mut u32,
    sf: &mut Datasize,
    op: &mut bool,
    imm19: &mut i32,
    rt: &mut RegisterId,
) -> bool {
    let insn = address.cast::<i32>().read();
    *sf = transmute((insn >> 31) & 1);
    *op = ((insn >> 24) & 0x1) != 0;
    *imm19 = (insn << 8) >> 13;
    *rt = (insn & 0x1f) as _;
    (insn & 0x7e000000) == 0x34000000
}

pub unsafe fn disassemble_conditional_branch_immediate(
    address: *mut u32,
    op01: &mut bool,
    imm19: &mut i32,
    condition: &mut Condition,
) -> bool {
    let insn = address.cast::<i32>().read();
    *op01 = (((insn >> 23) & 0x02) | ((insn >> 4) & 0x1)) != 0;
    *imm19 = (insn << 8) >> 13;
    *condition = transmute(insn & 0xf);
    (insn as u32 & 0xfe000000) == 0x54000000
}

pub unsafe fn disassemble_test_and_branch_immediate(
    address: *mut u32,
    op: &mut bool,
    bitnumber: &mut usize,
    imm14: &mut i32,
    rt: &mut RegisterId,
) -> bool {
    let insn = address.cast::<i32>().read();
    *op = ((insn >> 24) & 0x1) != 0;
    *imm14 = (insn << 13) >> 18;
    *bitnumber = (((insn >> 26) & 0x20) | ((insn >> 19) & 0x1f)) as usize;
    *rt = (insn & 0x1f) as _;
    (insn & 0x7e000000) == 0x36000000
}

pub unsafe fn disassemble_unconditional_branch_immediate(
    address: *mut u32,
    op: &mut bool,
    imm26: &mut i32,
) -> bool {
    let insn = address.cast::<i32>().read();
    *op = ((insn >> 31) & 1) != 0;
    *imm26 = (insn << 6) >> 6;
    (insn & 0x7c000000) == 0x14000000
}

pub fn xOrSp(reg: RegisterId) -> i32 {
    reg as _
}

pub fn xOrZr(reg: RegisterId) -> i32 {
    (reg as i32) & 31
}

pub fn xOrZrAsFPR(reg: RegisterId) -> i32 {
    xOrZr(reg)
}

pub fn xOrZrOrSp(use_zr: bool, reg: RegisterId) -> i32 {
    if use_zr {
        xOrZr(reg)
    } else {
        xOrSp(reg)
    }
}

pub fn add_subtract_extended_register(
    sf: Datasize,
    op: AddOp,
    s: SetFlags,
    rm: RegisterId,
    option: ExtendOp,
    imm3: i32,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let opt = 0;
    0x0b200000
        | (sf as i32) << 31
        | (op as i32) << 30
        | (s as i32) << 29
        | (opt as i32) << 22
        | xOrZr(rm) << 16
        | (option as i32) << 13
        | (imm3 & 0x7) << 10
        | xOrSp(rn) << 5
        | xOrZrOrSp(s as i32 != 0, rd)
}

pub fn add_subtract_immediate(
    sf: Datasize,
    op: AddOp,
    s: SetFlags,
    shift: i32,
    imm12: i32,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    0x11000000
        | (sf as i32) << 31
        | (op as i32) << 30
        | (s as i32) << 29
        | shift << 22
        | (imm12 & 0xfff) << 10
        | xOrSp(rn) << 5
        | xOrZrOrSp(s as i32 != 0, rd)
}

pub fn add_subtract_shifted_register(
    sf: Datasize,
    op: AddOp,
    s: SetFlags,
    shift: ShiftType,
    rm: RegisterId,
    imm6: i32,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    0x0b000000
        | (sf as i32) << 31
        | (op as i32) << 30
        | (s as i32) << 29
        | (shift as i32) << 22
        | xOrZr(rm) << 16
        | (imm6 & 0x3f) << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn add_subtract_with_carry(
    sf: Datasize,
    op: AddOp,
    s: SetFlags,
    rm: RegisterId,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let opcode2 = 0;
    0x1a000000
        | (sf as i32) << 31
        | (op as i32) << 30
        | (s as i32) << 29
        | xOrZr(rm) << 16
        | opcode2 << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn bitfield(
    sf: Datasize,
    opc: BitfieldOp,
    immr: i32,
    imms: i32,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    0x13000000
        | (sf as i32) << 31
        | (opc as i32) << 29
        | (N as i32) << 22
        | immr << 16
        | imms << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn compare_and_branch_immediate(sf: Datasize, op: bool, imm19: i32, rt: RegisterId) -> i32 {
    0x34000000 | (sf as i32) << 31 | (op as i32) << 24 | (imm19 & 0x7ffff) << 5 | xOrZr(rt)
}

pub fn conditional_branch_immediate(imm19: i32, cond: Condition) -> i32 {
    let o1 = 0;
    let o0 = 0;
    0x54000000 | o1 << 24 | (imm19 & 0x7ffff) << 5 | o0 << 4 | cond as i32
}

pub fn conditional_compare_immediate(
    sf: Datasize,
    op: AddOp,
    imm5: i32,
    cond: Condition,
    rn: RegisterId,
    vnzcv: i32,
) -> i32 {
    let S = 1;
    let o2 = 0;
    let o3 = 0;
    0x1a400800
        | (sf as i32) << 31
        | (op as i32) << 30
        | S << 29
        | (imm5 & 0x1f) << 16
        | (cond as i32) << 12
        | o2 << 10
        | xOrZr(rn) << 5
        | o3 << 4
        | vnzcv
}

pub fn conditional_compare_register(
    sf: Datasize,
    op: AddOp,
    rm: RegisterId,
    cond: Condition,
    rn: RegisterId,
    vnzcv: i32,
) -> i32 {
    let S = 1;
    let o2 = 0;
    let o3 = 0;
    0x1a400000
        | (sf as i32) << 31
        | (op as i32) << 30
        | S << 29
        | xOrZr(rm) << 16
        | (cond as i32) << 12
        | o2 << 10
        | xOrZr(rn) << 5
        | o3 << 4
        | vnzcv
}

pub fn conditional_select(
    sf: Datasize,
    op: bool,
    rm: RegisterId,
    cond: Condition,
    op2: bool,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let S = 0;
    0x1a800000
        | (sf as i32) << 31
        | (op as i32) << 30
        | S << 29
        | xOrZr(rm) << 16
        | (cond as i32) << 12
        | (op2 as i32) << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn data_processing_1source(
    sf: Datasize,
    opcode: DataOp1Source,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let S = 0;
    let opcode2 = 0;
    0x5ac00000
        | (sf as i32) << 31
        | S << 29
        | (opcode2 as i32) << 16
        | (opcode as i32) << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn data_processing_2source(
    sf: Datasize,
    rm: RegisterId,
    opcode: DataOp2Source,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let S = 0;
    0x1ac00000
        | (sf as i32) << 31
        | S << 29
        | xOrZr(rm) << 16
        | (opcode as i32) << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn data_processing_3source(
    sf: Datasize,
    opcode: DataOp3Source,
    rm: RegisterId,
    ra: RegisterId,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let op54 = (opcode as i32) >> 4;
    let op31 = (opcode as i32 >> 1) & 7;
    let op0 = opcode as i32 & 1;
    0x1b000000
        | (sf as i32) << 31
        | op54 << 29
        | op31 << 21
        | xOrZr(rm) << 16
        | op0 << 15
        | xOrZr(ra) << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn excepn_generation(opc: ExcepnOp, imm16: u16, LL: i32) -> i32 {
    let op2 = 0;
    (0xd4000000u32 | (opc as u32) << 21 | (imm16 as u32) << 5 | op2 << 2 | LL as u32) as i32
}

pub fn excepn_generation_imm_mask() -> i32 {
    let imm16 = u16::MAX as i32;
    imm16 << 5
}

pub fn extract(sf: Datasize, rm: RegisterId, imms: i32, rn: RegisterId, rd: RegisterId) -> i32 {
    let op21 = 0;
    let n = sf as i32;
    let o0 = 0;
    0x13800000
        | (sf as i32) << 31
        | op21 << 29
        | n << 22
        | o0 << 21
        | xOrZr(rm) << 16
        | imms << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn floating_point_immediate(typ: Datasize, imm8: i32, rd: RegisterId) -> i32 {
    let M = 0;
    let S = 0;
    let imm5 = 0;
    0x1e201000
        | M << 31
        | S << 29
        | (typ as i32) << 22
        | (imm8 & 0xff) << 13
        | imm5 << 5
        | rd as i32
}

pub fn floating_point_compare(
    typ: Datasize,
    rm: RegisterId,
    rn: RegisterId,
    opcode2: FPCmpOp,
) -> i32 {
    let M = 0;
    let S = 0;
    let op = 0;
    0x1e202000
        | M << 31
        | S << 29
        | (typ as i32) << 22
        | (rm as i32) << 16
        | op << 14
        | (rn as i32) << 5
        | opcode2 as i32
}

pub fn floating_point_conditional_compare(
    typ: Datasize,
    rm: RegisterId,
    cond: Condition,
    rn: RegisterId,
    op: FPCondCmpOp,
    vnzcv: i32,
) -> i32 {
    let M = 0;
    let S = 0;
    0x1e200400
        | M << 31
        | S << 29
        | (typ as i32) << 22
        | (rm as i32) << 16
        | (cond as i32) << 12
        | (rn as i32) << 5
        | (op as i32) << 4
        | vnzcv as i32
}

pub fn floating_point_conditional_select(
    typ: Datasize,
    rm: RegisterId,
    cond: Condition,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let M = 0;
    let S = 0;

    0x1e200c00
        | M << 31
        | S << 29
        | (typ as i32) << 22
        | (rm as i32) << 16
        | (cond as i32) << 12
        | (rn as i32) << 5
        | rd as i32
}

pub fn floating_point_integer_convertions(
    sf: Datasize,
    typ: Datasize,
    rmode_opcode: FPIntConvOp,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let S = 0;
    0x1e200000
        | (sf as i32) << 31
        | S << 29
        | (typ as i32) << 22
        | (rmode_opcode as i32) << 16
        | (rn as i32) << 5
        | rd as i32
}

pub fn floating_point_integer_convertions_fr(
    sf: Datasize,
    typ: Datasize,
    rmode_opcode: FPIntConvOp,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    floating_point_integer_convertions(sf, typ, rmode_opcode, rn, xOrZrAsFPR(rd) as _)
}

pub fn floating_point_integer_convertions_rr(
    sf: Datasize,
    typ: Datasize,
    rmode_opcode: FPIntConvOp,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    floating_point_integer_convertions_fr(sf, typ, rmode_opcode, xOrZrAsFPR(rn) as _, rd)
}

pub fn floating_point_data_processing_1source(
    typ: Datasize,
    opcode: FPDataOp1Source,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let m = 0;
    let s = 0;
    0x1e204000
        | m << 31
        | s << 29
        | (typ as i32) << 22
        | (opcode as i32) << 15
        | (rn as i32) << 5
        | (rd as i32)
}

pub fn floating_point_data_processing_2source(
    typ: Datasize,

    rm: RegisterId,
    opcode: FPDataOp2Source,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let m = 0;
    let s = 0;
    0x1e200800
        | m << 31
        | s << 29
        | (typ as i32) << 22
        | (rm as i32) << 16
        | (opcode as i32) << 12
        | (rn as i32) << 5
        | (rd as i32)
}

pub fn floating_point_data_processing_3source(
    typ: Datasize,
    o1: bool,
    rm: RegisterId,
    o2: AddOp,
    ra: RegisterId,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    let m = 0;
    let s = 0;
    0x1f000000
        | m << 31
        | s << 29
        | (typ as i32) << 22
        | (o1 as i32) << 21
        | (rm as i32) << 16
        | (o2 as i32) << 15
        | (ra as i32) << 10
        | (rn as i32) << 5
        | rd as i32
}

pub fn vector_processing_logical(
    u_and_size: SIMD3SameLogical,
    vm: RegisterId,
    vn: RegisterId,
    vd: RegisterId,
) -> i32 {
    let Q = 0;
    0xe200400
        | Q << 30
        | (u_and_size as i32) << 22
        | (vm as i32) << 16
        | (SIMD_LOGICAL_OP as i32) << 11
        | (vn as i32) << 5
        | (vd as i32)
}

pub fn load_register_literal(opc: u8, v: bool, imm19: i32, rt: RegisterId) -> i32 {
    0x18000000 | (opc as i32) << 30 | (v as i32) << 26 | (imm19 & 0x7ffff) << 5 | rt as i32
}

pub fn load_register_literal_r(opc: u8, v: bool, imm19: i32, rt: RegisterId) -> i32 {
    load_register_literal(opc, v, imm19, xOrZrAsFPR(rt) as _)
}

pub fn load_store_register_post_index(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    0x38000400
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm9 & 0x1ff) << 12
        | xOrSp(rn) << 5
        | rt as i32
}

pub fn load_store_register_post_index_r(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    load_store_register_post_index(size, v, opc, imm9, rn, xOrZrAsFPR(rt) as _)
}

pub fn load_store_register_pair_post_index(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    let immed_shift_amount = mem_pair_offset_shifted(v, size);
    let imm7 = immediate >> immed_shift_amount as i32;
    0x28800000
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm7 & 0x7f) << 15
        | (rt2 as i32) << 10
        | xOrSp(rn) << 5
        | (rt as i32)
}

pub fn load_store_register_pair_post_index_r(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    load_store_register_pair_post_index(
        size,
        v,
        opc,
        immediate,
        rn,
        xOrZrAsFPR(rt) as _,
        xOrZrAsFPR(rt2) as _,
    )
}

pub fn load_store_register_pre_index(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    0x38000c00
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm9 & 0x1ff) << 12
        | xOrSp(rn) << 5
        | rt as i32
}

pub fn load_store_register_pre_index_r(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    load_store_register_pre_index(size, v, opc, imm9, rn, xOrZrAsFPR(rt) as _)
}
pub fn load_store_register_pair_pre_index(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    let immed_shift_amount = mem_pair_offset_shifted(v, size);
    let imm7 = immediate >> immed_shift_amount as i32;
    0x29800000
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm7 & 0x7f) << 15
        | (rt2 as i32) << 10
        | xOrSp(rn) << 5
        | (rt as i32)
}

pub fn load_store_register_pair_pre_index_r(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    load_store_register_pair_pre_index(
        size,
        v,
        opc,
        immediate,
        rn,
        xOrZrAsFPR(rt) as _,
        xOrZrAsFPR(rt2) as _,
    )
}

pub fn load_store_register_pair_offset(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    let immed_shift_amount = mem_pair_offset_shifted(v, size);
    let imm7 = immediate >> immed_shift_amount as i32;
    0x29000000
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm7 & 0x7f) << 15
        | (rt2 as i32) << 10
        | xOrSp(rn) << 5
        | (rt as i32)
}

pub fn load_store_register_pair_offset_r(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    load_store_register_pair_offset(
        size,
        v,
        opc,
        immediate,
        rn,
        xOrZrAsFPR(rt) as _,
        xOrZrAsFPR(rt2) as _,
    )
}

pub fn load_store_register_pair_non_temporal(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    let immed_shift_amount = mem_pair_offset_shifted(v, size);
    let imm7 = immediate >> immed_shift_amount as i32;
    0x28000000
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm7 & 0x7f) << 15
        | (rt2 as i32) << 10
        | xOrSp(rn) << 5
        | (rt as i32)
}

pub fn load_store_register_pair_non_temporal_r(
    size: u8,
    v: bool,
    opc: MemOp,
    immediate: i32,
    rn: RegisterId,
    rt: RegisterId,
    rt2: RegisterId,
) -> i32 {
    load_store_register_pair_non_temporal(
        size,
        v,
        opc,
        immediate,
        rn,
        xOrZrAsFPR(rt) as _,
        xOrZrAsFPR(rt2) as _,
    )
}

pub fn load_store_register_offset(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    rm: RegisterId,
    option: ExtendOp,
    s: bool,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    0x38200800
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | xOrZr(rm) << 16
        | (option as i32) << 13
        | (s as i32) << 12
        | xOrSp(rn) << 5
        | rt as i32
}

pub fn load_store_register_offset_r(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    rm: RegisterId,
    option: ExtendOp,
    s: bool,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    load_store_register_offset(size, v, opc, rm, option, s, rn, xOrZrAsFPR(rt) as _)
}

pub fn load_store_register_unscaled_immediate(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    0x38000000
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm9 & 0x1ff) << 12
        | xOrSp(rn) << 5
        | rt as i32
}

pub fn load_store_register_unscaled_immediate_r(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    load_store_register_unscaled_immediate(size, v, opc, imm9, rn, xOrZrAsFPR(rt) as _)
}

pub fn load_store_register_unsigned_immediate(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    0x39000000
        | (size as i32) << 30
        | (v as i32) << 26
        | (opc as i32) << 22
        | (imm9 & 0xfff) << 10
        | xOrSp(rn) << 5
        | rt as i32
}

pub fn load_store_register_unsigned_immediate_r(
    size: MemOpSIZE,
    v: bool,
    opc: MemOp,
    imm9: i32,
    rn: RegisterId,
    rt: RegisterId,
) -> i32 {
    load_store_register_unsigned_immediate(size, v, opc, imm9, rn, xOrZrAsFPR(rt) as _)
}

pub fn logical_immediate(
    sf: Datasize,
    opc: LogicalOp,
    N_immr_imms: i32,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    0x12000000
        | (sf as i32) << 31
        | (opc as i32) << 29
        | N_immr_imms << 10
        | xOrZr(rn) << 5
        | xOrZrOrSp(opc == LogicalOp::ANDS, rd)
}

pub fn logical_shifted_register(
    sf: Datasize,
    opc: LogicalOp,
    shift: ShiftType,
    n: bool,
    rm: RegisterId,
    imm6: i32,
    rn: RegisterId,
    rd: RegisterId,
) -> i32 {
    0x0a000000
        | (sf as i32) << 31
        | (opc as i32) << 29
        | (shift as i32) << 22
        | (n as i32) << 21
        | xOrZr(rm) << 16
        | (imm6 & 0x3f) << 10
        | xOrZr(rn) << 5
        | xOrZr(rd)
}

pub fn move_wide_immediate(
    sf: Datasize,
    opc: MoveWideOp,
    hw: i32,
    imm16: u16,
    rd: RegisterId,
) -> i32 {
    0x12800000 | (sf as i32) << 31 | (opc as i32) << 29 | hw << 21 | (imm16 as i32) << 5 | xOrZr(rd)
}

pub fn unconditional_branch_immediate(op: bool, imm26: i32) -> i32 {
    0x14000000 | (op as i32) << 31 | (imm26 & 0x3ffffff)
}

pub fn pc_relative(op: bool, imm21: i32, rd: RegisterId) -> i32 {
    let immlo = imm21 & 3;
    let op = op as i32;
    let immhi = (imm21 >> 2) & 0x7ffff;
    0x10000000 | op << 31 | immlo << 29 | immhi << 5 | xOrZr(rd)
}

pub fn system(l: bool, op0: i32, op1: i32, crn: i32, crm: i32, op2: i32, rt: RegisterId) -> i32 {
    (0xd5000000u32 as i32)
        | (l as i32) << 21
        | op0 << 19
        | op1 << 16
        | crn << 12
        | crm << 8
        | op2 << 5
        | xOrZr(rt)
}

pub fn data_cache_zero_virtual_address(rt: RegisterId) -> i32 {
    system(false, 1, 0x3, 0x7, 0x4, 0x1, rt)
}

pub fn hint_pseudo(imm: i32) -> i32 {
    system(false, 0, 3, 2, (imm >> 3) & 0xf, imm & 0x7, zr)
}
pub fn nop_pseudo() -> i32 {
    hint_pseudo(0)
}

pub fn test_and_branch_immediate(op: bool, b50: i32, imm14: i32, rt: RegisterId) -> i32 {
    let b5 = b50 >> 5;
    let b40 = b50 & 0x1f;
    let op = op as i32;

    0x36000000 | b5 << 31 | op << 24 | b40 << 19 | (imm14 & 0x3fff) << 5 | xOrZr(rt)
}

pub fn unconditional_branch_register(opc: BranchType, rn: RegisterId) -> i32 {
    let op2 = 0x1f;
    let op3 = 0;
    let op4 = 0;
    let opc = opc as i32;
    (0xd6000000u32 as i32) | opc << 21 | op2 << 16 | op3 << 10 | xOrZr(rn) << 5 | op4
}

pub fn exotic_load(
    size: MemOpSIZE,
    fence: ExoticLoadFence,
    atomic: ExoticLoadAtomic,
    dst: RegisterId,
    src: RegisterId,
) -> i32 {
    let size = size as i32;
    let fence = fence as i32;
    let atomic = atomic as i32;
    let dst = dst as i32;
    let src = src as i32;
    0x085f7c00 | size << 30 | fence << 15 | atomic << 23 | src << 5 | dst
}

pub fn store_release(size: MemOpSIZE, src: RegisterId, dst: RegisterId) -> i32 {
    let size = size as i32;
    let dst = dst as i32;
    let src = src as i32;
    0x089ffc00 | size << 30 | dst << 5 | src
}

pub fn exotic_store(
    size: MemOpSIZE,
    fence: ExoticStoreFence,
    result: RegisterId,
    src: RegisterId,
    dst: RegisterId,
) -> i32 {
    let size = size as i32;
    let fence = fence as i32;
    let result = result as i32;
    let src = src as i32;
    let dst = dst as i32;
    0x08007c00 | size << 30 | result << 16 | fence << 15 | dst << 5 | src
}

pub fn fjcvtzs(dn: RegisterId, rd: RegisterId) -> i32 {
    let dn = dn as i32;
    let rd = rd as i32;
    0x1e7e0000 | (dn << 5) | rd
}

pub const MAX_POINTER_BITS: usize = 48;

pub fn invert(cond: Condition) -> Condition {
    let x = cond as i32 ^ 1;
    unsafe { transmute(x) }
}

pub unsafe fn set_pointer(address: *mut u32, value_ptr: *mut u8, rd: RegisterId, _flush: bool) {
    let value = value_ptr as usize;
    let buffer = [
        move_wide_immediate(
            Datasize::D64,
            MoveWideOp::Z,
            0,
            get_half_word(value as _, 0) as _,
            rd,
        ),
        move_wide_immediate(
            Datasize::D64,
            MoveWideOp::K,
            0,
            get_half_word(value as _, 1) as _,
            rd,
        ),
        move_wide_immediate(
            Datasize::D64,
            MoveWideOp::K,
            0,
            get_half_word(value as _, 3) as _,
            rd,
        ),
    ];

    core::ptr::copy_nonoverlapping(buffer.as_ptr(), address.cast(), 3);
}

pub unsafe fn repatch_int32(at: *mut u8, value: i32) {
    let address = at;
    let mut sf = Datasize::D16;
    let mut opc = MoveWideOp::K;
    let mut hw = 0;
    let mut imm16 = 0;
    let mut rd = zr;
    let _ = disassemble_mov_with_imediate(
        address.cast(),
        &mut sf,
        &mut opc,
        &mut hw,
        &mut imm16,
        &mut rd,
    );
    let mut buffer = [0; 2];
    if value >= 0 {
        buffer[0] = move_wide_immediate(
            Datasize::D32,
            MoveWideOp::Z,
            0,
            get_half_word32(value as _, 0) as _,
            rd,
        );
        buffer[1] = move_wide_immediate(
            Datasize::D32,
            MoveWideOp::K,
            1,
            get_half_word32(value as _, 1) as _,
            rd,
        );
    } else {
        buffer[0] = move_wide_immediate(
            Datasize::D32,
            MoveWideOp::N,
            0,
            !get_half_word32(value as _, 0) as _,
            rd,
        );
        buffer[1] = move_wide_immediate(
            Datasize::D32,
            MoveWideOp::K,
            1,
            get_half_word32(value as _, 1) as _,
            rd,
        );
    }

    core::ptr::copy_nonoverlapping(buffer.as_ptr(), at.cast(), 2);
}

pub unsafe fn read_pointer(at: *mut u8) -> *mut u8 {
    let address = at.cast::<u32>();

    let mut sf = Datasize::D16;
    let mut opc = MoveWideOp::K;
    let mut hw = 0;
    let mut imm16 = 0;
    let mut rd = zr;
    let mut rdfirst = zr;
    let _ = disassemble_mov_with_imediate(
        address,
        &mut sf,
        &mut opc,
        &mut hw,
        &mut imm16,
        &mut rdfirst,
    );
    let mut result = imm16 as usize;

    let _ = disassemble_mov_with_imediate(
        address.add(1),
        &mut sf,
        &mut opc,
        &mut hw,
        &mut imm16,
        &mut rd,
    );

    result |= (imm16 as usize) << 16;

    #[cfg(target_pointer_width = "64")]
    {
        let _ = disassemble_mov_with_imediate(
            address.add(2),
            &mut sf,
            &mut opc,
            &mut hw,
            &mut imm16,
            &mut rd,
        );

        result |= (imm16 as usize) << 32;
    }
    result as _
}

pub unsafe fn read_call_target(from: *mut u32) -> *mut u8 {
    read_pointer(
        from.sub(if cfg!(target_pointer_width = "64") {
            4
        } else {
            3
        })
        .cast(),
    )
    .cast()
}

pub fn jump_size_delta(t: JumpType, l: JumpLinkType) -> i32 {
    jump_enum_size(t as i32) - jump_enum_size(l as i32)
}

pub const fn jump_enum_size(x: i32) -> i32 {
    x >> 4
}

pub fn can_compact(j: JumpType) -> bool {
    match j {
        JumpType::NoCondition
        | JumpType::Condition
        | JumpType::CompareAndBranch
        | JumpType::TestBit => true,
        _ => false,
    }
}

pub unsafe fn compute_jump_type(t: JumpType, from: *const u8, to: *const u8) -> JumpLinkType {
    use JumpType::*;
    match t {
        Fixed => JumpLinkType::Invalid,
        NoConditionFixedSize => JumpLinkType::NoCondition,
        ConditionFixedSize => JumpLinkType::Condition,
        CompareAndBranchFixedSize => JumpLinkType::CompareAndBranch,
        TestBitFixedSize => JumpLinkType::TestBit,
        NoCondition => JumpLinkType::NoCondition,
        Condition => {
            let relative = to as isize - from as isize;
            if is_int!(21, relative) {
                return JumpLinkType::ConditionDirect;
            }

            JumpLinkType::Condition
        }
        CompareAndBranch => {
            let relative = to as isize - from as isize;
            if is_int!(21, relative) {
                return JumpLinkType::CompareAndBranchDirect;
            }

            JumpLinkType::CompareAndBranch
        }
        TestBit => {
            let relative = to as isize - from as isize;
            if is_int!(14, relative) {
                return JumpLinkType::TestBitDirect;
            }

            JumpLinkType::TestBit
        }
    }
}

pub fn can_emit_jump(from: *mut u8, to: *mut u8) -> bool {
    let diff = from as isize - to as isize;
    is_int!(26, diff)
}

pub unsafe fn link_pointer(address: *mut u32, value_ptr: *mut u8) {
    let mut sf = Datasize::D16;
    let mut opc = MoveWideOp::K;
    let mut hw = 0;
    let mut imm16 = 0;
    let mut rd = zr;
    let _ = disassemble_mov_with_imediate(
        address.cast(),
        &mut sf,
        &mut opc,
        &mut hw,
        &mut imm16,
        &mut rd,
    );
    set_pointer(address, value_ptr, rd, false);
}

pub unsafe fn repatch_pointer(at: *mut u8, value_ptr: *mut u8) {
    link_pointer(at.cast(), value_ptr);
}

pub unsafe fn link_jump_or_call(
    typ: BranchType,
    from: *mut u32,
    from_instruction: *mut u32,
    to: *mut u8,
) {
    let mut link = false;
    let mut imm26 = 0;
    let _ = disassemble_unconditional_branch_immediate(from.cast(), &mut link, &mut imm26)
        || disassemble_nop(from.cast());

    let is_call = typ == BranchType::CALL;

    let offset = (to as isize - from_instruction as isize) >> 2;

    let insn = unconditional_branch_immediate(is_call, offset as _);

    std::ptr::copy_nonoverlapping(&insn, from.cast(), 1);
}

pub unsafe fn link_compare_and_branch(
    typ: BranchTargetType,
    condition: Condition,
    is64bit: bool,
    rt: RegisterId,
    from: *mut u32,
    from_instruction: *mut u32,
    to: *mut u8,
) {
    let offset = (to as isize - from_instruction as isize) >> 2;

    let use_direct = is_int!(19, offset);
    if use_direct || typ == BranchTargetType::Direct {
        let insn = compare_and_branch_immediate(
            if is64bit {
                Datasize::D64
            } else {
                Datasize::D32
            },
            condition == Condition::NE,
            offset as _,
            rt,
        );
        std::ptr::copy_nonoverlapping(&insn, from.cast(), 1);
        if typ == BranchTargetType::Indirect {
            let insn = nop_pseudo();
            std::ptr::copy_nonoverlapping(&insn, from.add(1).cast(), 1);
        }
    } else {
        let insn = compare_and_branch_immediate(
            if is64bit {
                Datasize::D64
            } else {
                Datasize::D32
            },
            invert(condition) == Condition::NE,
            2,
            rt,
        );
        std::ptr::copy_nonoverlapping(&insn, from.cast(), 1);
        link_jump_or_call(BranchType::JMP, from.add(1), from_instruction.add(1), to);
    }
}

pub unsafe fn link_conditional_branch(
    typ: BranchTargetType,
    condition: Condition,
    from: *mut u32,
    from_instruction: *mut u32,
    to: *mut u8,
) {
    let offset = (to as isize - from_instruction as isize) >> 2;

    let use_direct = is_int!(19, offset);
    if use_direct || typ == BranchTargetType::Direct {
        let insn = conditional_branch_immediate(offset as _, condition);
        std::ptr::copy_nonoverlapping(&insn, from.cast(), 1);
        if typ == BranchTargetType::Indirect {
            let insn = nop_pseudo();
            std::ptr::copy_nonoverlapping(&insn, from.add(1).cast(), 1);
        }
    } else {
        let insn = conditional_branch_immediate(2, invert(condition));
        std::ptr::copy_nonoverlapping(&insn, from.cast(), 1);
        link_jump_or_call(BranchType::JMP, from.add(1), from_instruction.add(1), to);
    }
}

pub unsafe fn link_test_and_branch(
    typ: BranchTargetType,
    condition: Condition,
    bitnumber: usize,
    rt: RegisterId,
    from: *mut u32,
    from_instruction: *mut u32,
    to: *mut u8,
) {
    let offset = (to as isize - from_instruction as isize) >> 2;

    let use_direct = is_int!(14, offset);

    if use_direct || typ == BranchTargetType::Direct {
        let insn =
            test_and_branch_immediate(condition == Condition::NE, bitnumber as _, offset as _, rt);

        std::ptr::copy_nonoverlapping(&insn, from.cast(), 1);

        if typ == BranchTargetType::Indirect {
            let insn = nop_pseudo();

            std::ptr::copy_nonoverlapping(&insn, from.add(1).cast(), 1);
        }
    } else {
        let insn =
            test_and_branch_immediate(invert(condition) == Condition::NE, bitnumber as _, 2, rt);
        std::ptr::copy_nonoverlapping(&insn, from.cast(), 1);
        link_jump_or_call(BranchType::JMP, from, from_instruction, to)
    }
}

pub unsafe fn relink_jump(from: *mut u8, to: *mut u8) {
    relink_jump_or_call(BranchType::JMP, from.cast(), from.cast(), to);
}

pub unsafe fn relink_call(from: *mut u8, to: *mut u8) {
    relink_jump_or_call(BranchType::CALL, from.sub(1).cast(), from.sub(1).cast(), to);
}

pub unsafe fn relink_tail_call(from: *mut u8, to: *mut u8) {
    relink_jump(from, to)
}

pub unsafe fn replace_with_address_computation(at: *mut u8) {
    let mut size = MemOpSIZE::S16;
    let mut v = false;
    let mut opc = MemOp::LOAD;
    let mut imm12 = 0;
    let mut rn = 0;
    let mut rt = 0;
    if disassemble_load_store_register_unsigned_immediate(
        at.cast(),
        &mut size,
        &mut v,
        &mut opc,
        &mut imm12,
        &mut rn,
        &mut rt,
    ) {
        let insn = add_subtract_immediate(
            Datasize::D64,
            AddOp::ADD,
            SetFlags::DontSet,
            0,
            imm12 * size_of::<usize>() as i32,
            rn,
            rt,
        );

        std::ptr::copy_nonoverlapping(&insn, at.cast(), 1);
    }
}

pub unsafe fn replace_with_load(at: *mut u8) {
    let mut size = Datasize::D32;
    //let mut v = false;

    let mut imm12 = 0;
    let mut rn = 0;
    let mut rd = 0;
    let mut op = AddOp::ADD;
    let mut s = SetFlags::DontSet;
    let mut shift = 0;
    if disassemble_add_subtract_immediate(
        at.cast(),
        &mut size,
        &mut op,
        &mut s,
        &mut shift,
        &mut imm12,
        &mut rn,
        &mut rd,
    ) {
        let insn = load_store_register_unsigned_immediate(
            MemOpSIZE::S64,
            false,
            MemOp::LOAD,
            encode_positive_immediate(64, imm12 as _),
            rn,
            rd,
        );
        std::ptr::copy_nonoverlapping(&insn, at.cast(), 1);
    }
}

pub unsafe fn replace_with_jump(at: *mut u8, to: *mut u8) {
    let offset = (to as isize - at as isize) >> 2;
    let insn = unconditional_branch_immediate(false, offset as _);
    std::ptr::copy_nonoverlapping(&insn, at.cast(), 1);
}

pub unsafe fn repalce_with_vm_halt(at: *mut u8) {
    let insn = data_cache_zero_virtual_address(zr);
    std::ptr::copy_nonoverlapping(&insn, at.cast(), 1);
}

impl LinkRecord {
    pub fn compute_jump_type(&mut self, from: *const u8, to: *const u8) -> JumpLinkType {
        unsafe {
            self.link = compute_jump_type(self.typ, from, to);
        }
        self.link
    }
}

pub const fn encode_positive_immediate(datasize: i32, pimm: usize) -> i32 {
    (pimm / (datasize as usize / 8)) as i32
}

impl ARM64Assembler {
    pub unsafe fn link_pointer(code: *mut u8, at: AssemblerLabel, value_ptr: *mut u8) {
        link_pointer(address_of(code, at), value_ptr)
    }

    pub unsafe fn link_call(code: *mut u8, from: AssemblerLabel, to: *mut u8) {
        link_jump_or_call(
            BranchType::CALL,
            address_of(code, from).sub(1),
            address_of(code, from).sub(1),
            to,
        )
    }

    pub unsafe fn link_jump(code: *mut u8, from: AssemblerLabel, to: *mut u8) {
        link_jump_or_call(
            BranchType::JMP,
            address_of(code, from).sub(1),
            address_of(code, from).sub(1),
            to,
        )
    }
}

impl Assembler for ARM64Assembler {
    fn link_jump(code: *mut u8, jump: Jump, target: *mut u8) {
        unsafe {
            ARM64Assembler::link_jump(code, jump.label, target);
        }
    }

    fn get_difference_between_labels(a: AssemblerLabel, b: AssemblerLabel) -> isize {
        a.offset() as isize - b.offset() as isize
    }

    fn link_pointer(code: *mut u8, label: AssemblerLabel, value: *mut u8) {
        unsafe { ARM64Assembler::link_pointer(code, label, value) }
    }

    fn get_linker_address(code: *mut u8, label: AssemblerLabel) -> *mut u8 {
        ARM64Assembler::get_relocated_address(code, label)
    }

    fn read_pointer(at: *mut u8) -> *mut u8 {
        unsafe { read_pointer(at) }
    }

    fn get_linker_call_return_offset(label: AssemblerLabel) -> usize {
        label.offset() as _
    }

    fn relink_call(code: *mut u8, destination: *mut u8) {
        unsafe { relink_call(code, destination) }
    }

    fn relink_tail_call(code: *mut u8, destination: *mut u8) {
        unsafe { relink_tail_call(code, destination) }
    }

    fn repatch_int32(at: *mut u8, value: i32) {
        unsafe { repatch_int32(at, value) }
    }

    fn repatch_pointer(at: *mut u8, value: *mut u8) {
        unsafe { repatch_pointer(at, value) }
    }

    fn repatch_jump(jump: *mut u8, destination: *mut u8) {
        unsafe { relink_jump(jump, destination) }
    }

    fn replace_with_load(label: *mut u8) {
        unsafe { replace_with_load(label) }
    }

    fn replace_with_address_computation(label: *mut u8) {
        unsafe { replace_with_address_computation(label) }
    }
}

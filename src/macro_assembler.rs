use super::*;
use crate::{assembler_buffer::AssemblerLabel, link_buffer::LinkBuffer};
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum CallFlags {
    None = 0x0,
    Linkable = 0x1,
    Near = 0x2,
    Tail = 0x4,
    LinkableNear = Self::Linkable as u8 | Self::Near as u8,
    LinkableNearTail = Self::Linkable as u8 | Self::Near as u8 | Self::Tail as u8,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i8)]
pub enum Scale {
    One,
    Two,
    Four,
    Eight,
}

#[cfg(target_pointer_width = "64")]
pub const SCALE_PTR: Scale = Scale::Eight;
#[cfg(target_pointer_width = "32")]
pub const SCALE_PTR: Scale = Scale::Four;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i8)]
pub enum Extend {
    ZExt32,
    SExt32,
    None,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address {
    pub base: RegisterId,
    pub offset: i32,
}

impl Address {
    pub const fn new(base: RegisterId, offset: i32) -> Self {
        Self { base, offset }
    }

    pub const fn with_offset(self, offset: i32) -> Self {
        Self {
            offset: self.offset + offset,
            ..self
        }
    }

    pub fn with_swapped_register(self, left: RegisterId, right: RegisterId) -> Self {
        Self {
            base: with_swapped_register(self.base, left, right),
            ..self
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExtededAddress {
    pub base: RegisterId,
    pub offset: isize,
}

impl ExtededAddress {
    pub const fn new(base: RegisterId, offset: isize) -> Self {
        Self { base, offset }
    }

    pub const fn with_offset(self, offset: isize) -> Self {
        Self {
            offset: self.offset + offset,
            ..self
        }
    }

    pub fn with_swapped_register(self, left: RegisterId, right: RegisterId) -> Self {
        Self {
            base: with_swapped_register(self.base, left, right),
            ..self
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BaseIndex {
    pub base: RegisterId,
    pub index: RegisterId,
    pub scale: Scale,
    pub offset: i32,
    pub extend: Extend,
}

impl BaseIndex {
    pub const fn new(base: RegisterId, index: RegisterId, scale: Scale, offset: i32) -> Self {
        Self {
            base,
            index,
            scale,
            offset,
            extend: Extend::None,
        }
    }

    pub const fn with_offset(self, offset: i32) -> Self {
        Self {
            offset: self.offset + offset,
            ..self
        }
    }

    pub fn with_swapped_register(self, left: RegisterId, right: RegisterId) -> Self {
        Self {
            base: with_swapped_register(self.base, left, right),
            index: with_swapped_register(self.index, left, right),
            ..self
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PreIndexAddress {
    pub base: RegisterId,
    pub index: i32,
}

impl PreIndexAddress {
    pub const fn new(base: RegisterId, index: i32) -> Self {
        Self { base, index }
    }

    pub const fn with_offset(self, index: i32) -> Self {
        Self {
            index: self.index + index,
            ..self
        }
    }

    pub fn with_swapped_register(self, left: RegisterId, right: RegisterId) -> Self {
        Self {
            base: with_swapped_register(self.base, left, right),
            ..self
        }
    }
}

pub struct PostIndexAddress {
    pub base: RegisterId,
    pub index: i32,
}

impl PostIndexAddress {
    pub const fn new(base: RegisterId, index: i32) -> Self {
        Self { base, index }
    }

    pub const fn with_offset(self, index: i32) -> Self {
        Self {
            index: self.index + index,
            ..self
        }
    }

    pub fn with_swapped_register(self, left: RegisterId, right: RegisterId) -> Self {
        Self {
            base: with_swapped_register(self.base, left, right),
            ..self
        }
    }
}

pub struct AbsoluteAddress {
    pub ptr: *mut u8,
}

impl AbsoluteAddress {
    pub const fn new(ptr: *mut u8) -> Self {
        Self { ptr }
    }
}

pub struct ExtendedAddress {
    pub base: RegisterId,
    pub offset: isize,
}

impl ExtendedAddress {
    pub const fn new(base: RegisterId, offset: isize) -> Self {
        Self { base, offset }
    }
}

pub struct ImmPtr(pub *mut u8);

impl ImmPtr {
    pub const fn new(ptr: *mut u8) -> Self {
        Self(ptr)
    }

    pub fn as_isize(&self) -> isize {
        self.0 as isize
    }
}

pub struct Imm32(pub i32);

impl Imm32 {
    pub const fn new(value: i32) -> Self {
        Self(value)
    }
}

pub struct Imm64(pub i64);

impl Imm64 {
    pub const fn new(value: i64) -> Self {
        Self(value)
    }
}

pub fn with_swapped_register(
    original: RegisterId,
    left: RegisterId,
    right: RegisterId,
) -> RegisterId {
    if original == left {
        right
    } else if original == right {
        left
    } else {
        original
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Call {
    pub label: AssemblerLabel,
    pub(crate) flags: CallFlags,
}

impl Call {
    pub const fn new(label: AssemblerLabel, flags: CallFlags) -> Self {
        Self { label, flags }
    }

    pub const fn none() -> Self {
        Self {
            label: AssemblerLabel::new(0),
            flags: CallFlags::None,
        }
    }

    pub const fn is_flag_set(self, flag: CallFlags) -> bool {
        (self.flags as u8 & flag as u8) != 0
    }

    pub const fn from_tail_jump(jump: Jump) -> Self {
        Self::new(jump.label, CallFlags::Linkable)
    }
}

/// A Label records a point in the generated instruction stream, typically such that
/// it may be used as a destination for a jump.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Label {
    pub(crate) label: AssemblerLabel,
}

impl Label {
    pub fn new<T: MacroAssembler>(assembler: &mut T) -> Self {
        Label {
            label: assembler.raw_label(),
        }
    }

    pub const fn is_set(&self) -> bool {
        self.label.is_set()
    }

    pub const fn label(&self) -> AssemblerLabel {
        self.label
    }
}
/// A ConvertibleLoadLabel records a loadPtr instruction that can be patched to an addPtr
/// so that:
///
/// loadPtr(Address(a, i), b)
///
/// becomes:
///
/// addPtr(TrustedImmPtr(i), a, b)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct ConvertibleLoadLabel {
    pub(crate) label: AssemblerLabel,
}

impl ConvertibleLoadLabel {
    pub fn new<T: MacroAssembler>(assembler: &mut T) -> Self {
        ConvertibleLoadLabel {
            label: assembler.raw_label(),
        }
    }
    pub const fn label(&self) -> AssemblerLabel {
        self.label
    }

    pub const fn is_set(&self) -> bool {
        self.label.is_set()
    }
}

/// A DataLabelPtr is used to refer to a location in the code containing a pointer to be
/// patched after the code has been generated.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct DataLabelPtr {
    pub(crate) label: AssemblerLabel,
}

impl DataLabelPtr {
    pub fn new<T: MacroAssembler>(assembler: &mut T) -> Self {
        DataLabelPtr {
            label: assembler.raw_label(),
        }
    }
    pub const fn label(&self) -> AssemblerLabel {
        self.label
    }
    pub const fn is_set(&self) -> bool {
        self.label.is_set()
    }
}
/// A DataLabel32 is used to refer to a location in the code containing a 32-bit constant to be
/// patched after the code has been generated.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct DataLabel32 {
    pub(crate) label: AssemblerLabel,
}

impl DataLabel32 {
    pub fn new<T: MacroAssembler>(assembler: &mut T) -> Self {
        DataLabel32 {
            label: assembler.raw_label(),
        }
    }
    pub const fn label(&self) -> AssemblerLabel {
        self.label
    }
    pub const fn is_set(&self) -> bool {
        self.label.is_set()
    }
}
/// A DataLabelCompact is used to refer to a location in the code containing a
/// compact immediate to be patched after the code has been generated.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct DataLabelCompact {
    pub(crate) label: AssemblerLabel,
}

impl DataLabelCompact {
    pub fn new<T: MacroAssembler>(assembler: &mut T) -> Self {
        DataLabelCompact {
            label: assembler.raw_label(),
        }
    }
    pub const fn label(&self) -> AssemblerLabel {
        self.label
    }
    pub const fn is_set(&self) -> bool {
        self.label.is_set()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Jump {
    pub(crate) label: AssemblerLabel,
    #[cfg(target_arch = "aarch64")]
    pub(crate) typ: JumpType,
    #[cfg(target_arch = "aarch64")]
    pub(crate) cond: Condition,
    #[cfg(target_arch = "aarch64")]
    pub(crate) bitnumber: usize,
    #[cfg(target_arch = "aarch64")]
    pub(crate) is64bit: bool,
    #[cfg(target_arch = "aarch64")]
    pub(crate) compare_register: RegisterId,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PatchableJump(pub(crate) Jump);

impl Deref for PatchableJump {
    type Target = Jump;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PatchableJump {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct JumpList {
    jumps: Vec<Jump>,
}

impl JumpList {
    pub fn new() -> Self {
        Self { jumps: vec![] }
    }

    pub fn link<T: MacroAssembler>(&self, masm: &mut T) {
        for i in 0..self.jumps.len() {
            masm.link_jump_(self.jumps[i]);
        }
    }

    pub fn link_to<T: MacroAssembler>(&self, label: Label, masm: &mut T) {
        for i in 0..self.jumps.len() {
            masm.link_to(self.jumps[i], label);
        }
    }

    pub fn append(&mut self, j: Jump) {
        self.jumps.push(j);
    }

    pub fn append_other(&mut self, this: &mut Self) {
        self.jumps.append(&mut this.jumps);
    }

    pub fn is_empty(&self) -> bool {
        self.jumps.is_empty()
    }

    pub fn clear(&mut self) {
        self.jumps.clear();
    }

    pub fn jumps(&self) -> &[Jump] {
        &self.jumps
    }

    pub fn jumps_mut(&mut self) -> &mut Vec<Jump> {
        &mut self.jumps
    }
}

pub trait Assembler {
    fn get_difference_between_labels(a: AssemblerLabel, b: AssemblerLabel) -> isize;
    fn link_jump(code: *mut u8, jump: Jump, target: *mut u8);
    fn link_pointer(code: *mut u8, label: AssemblerLabel, value: *mut u8);

    fn get_linker_address(code: *mut u8, label: AssemblerLabel) -> *mut u8;

    fn get_linker_call_return_offset(label: AssemblerLabel) -> usize;

    fn repatch_jump(jump: *mut u8, destination: *mut u8);
    fn relink_tail_call(code: *mut u8, destination: *mut u8);
    fn relink_call(code: *mut u8, destination: *mut u8);

    fn repatch_near_call(tail: bool, near_call: *mut u8, destination: *mut u8) {
        if tail {
            Self::relink_tail_call(near_call, destination);
        } else {
            Self::relink_call(near_call, destination);
        }
    }

    fn repatch_int32(at: *mut u8, value: i32);

    fn repatch_pointer(at: *mut u8, value: *mut u8);

    fn read_pointer(at: *mut u8) -> *mut u8;

    fn replace_with_load(label: *mut u8);
    fn replace_with_address_computation(label: *mut u8);
}

pub trait MacroAssembler: Sized {
    type GR;
    type FR;
    type Call: Copy + Clone + PartialEq + Eq + Debug = Call;
    type AssemblerType;
    fn first_register() -> RegisterId;
    fn last_register() -> RegisterId;
    fn number_of_registers() -> usize;

    fn first_sp_register() -> RegisterId;
    fn last_sp_register() -> RegisterId;
    fn number_of_sp_registers() -> usize;

    fn first_fp_register() -> RegisterId;
    fn last_fp_register() -> RegisterId;
    fn number_of_fp_registers() -> usize;

    fn link_call(code: *mut u8, call: Call, function: *mut u8);
    fn link_jump(code: *mut u8, from: AssemblerLabel, to: *mut u8);
    fn link_pointer(code: *mut u8, at: AssemblerLabel, value_ptr: *mut u8);
    fn compute_jump_type(typ: JumpType, from: *const u8, to: *const u8) -> JumpLinkType;
    fn jump_size_delta(typ: JumpType, link: JumpLinkType) -> i32;
    fn can_compact(typ: JumpType) -> bool;
    fn get_call_return_offset(label: AssemblerLabel) -> usize;

    fn get_linker_call_return_offset(call: Call) -> usize {
        Self::get_call_return_offset(call.label)
    }

    fn raw_label(&mut self) -> AssemblerLabel;

    fn label(&mut self) -> Label {
        Label::new(self)
    }
    fn watchpoint_label(&mut self) -> Label {
        self.label()
    }

    fn pad_before_patch(&mut self) {
        self.label();
    }

    fn align(&mut self) -> Label;

    fn link_jump_(&mut self, j: Jump);
    fn link_to(&mut self, j: Jump, label: Label);

    fn clear_temp_register_valid(&mut self, r: usize) {
        let _ = r;
    }

    fn is_temp_register_valid(&self, r: usize) -> bool {
        let _ = r;
        false
    }

    fn set_temp_register_valid(&mut self, r: usize) {
        let _ = r;
    }

    fn invalidate_all_temp_registers(&mut self) {}
}

pub struct AbstractMacroAssembler<ASM> {
    assembler: ASM,
}

#[derive(Copy, Clone)]
pub struct CachedTempRegister {
    reg: RegisterId,
    value: isize,
    valid_bit: usize,
}

impl CachedTempRegister {
    pub fn new(reg: RegisterId) -> Self {
        Self {
            reg,
            value: 0,
            valid_bit: 1 << reg as usize,
        }
    }
    pub fn register_id_invalidate(&mut self, masm: &mut impl MacroAssembler) -> RegisterId {
        self.invalidate(masm);
        self.reg
    }

    pub fn register_id_no_invalidate(&mut self) -> RegisterId {
        self.reg
    }

    pub fn value(&self, masm: &impl MacroAssembler, value: &mut isize) -> bool {
        *value = self.value;
        masm.is_temp_register_valid(self.valid_bit)
    }

    pub fn set_value(&mut self, masm: &mut impl MacroAssembler, value: isize) {
        self.value = value;
        masm.set_temp_register_valid(self.valid_bit);
    }

    pub fn invalidate(&mut self, masm: &mut impl MacroAssembler) {
        masm.clear_temp_register_valid(self.valid_bit);
    }
}

#[cfg(target_arch = "aarch64")]
pub mod arm64;

#[cfg(target_arch = "aarch64")]
pub use arm64::*;

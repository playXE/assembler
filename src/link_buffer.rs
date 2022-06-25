use std::marker::PhantomData;

use crate::macro_assembler::MacroAssembler;

/// LinkBuffer
///
///
/// This struct assists in linking code generated by the macro assembler, once code generation has been completed,
/// and the code has been copied to its final location in memory. At this time pointers to labels within the code
/// may be resolved, and relative offsets to external addresses may be fixed.
///
/// Specifically:
///   * Jump objects may be linked to external targets,
///   * The address of Jump objects may taken, such that it can later be relinked.
///   * The return address of a Call may be acquired.
///   * The address of a Label pointing into the code may be resolved.
///   * The value referenced by a DataLabel may be set.
pub struct LinkBuffer<Masm: MacroAssembler> {
    marker: PhantomData<Masm>,
}
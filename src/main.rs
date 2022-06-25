use capstone::{arch::arm64::ArchMode, prelude::*};
use masm::arm64::*;

fn main() {
    let mut asm = ARM64Assembler::new();

    asm.mov(64, x0, x2);
    let mut cs = Capstone::new().arm64().mode(ArchMode::Arm).build().unwrap();
    cs.set_skipdata(true).unwrap();

    let insns = cs
        .disasm_all(
            &asm.buffer().data().as_slice()[0..asm.buffer().code_size()],
            0x0,
        )
        .unwrap();
    for ins in insns.iter() {
        println!("{}", ins);
    }
}

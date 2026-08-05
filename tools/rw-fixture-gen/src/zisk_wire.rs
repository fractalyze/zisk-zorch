//! Rust mirror of the consumer-side ZisK wire formats (riscv-witness7).
//!
//! Single source of truth for byte layout lives on the consumer:
//!   * ZROM   → `riscv_witness/checkpoint/zisk/zisk_rom_io.cpp`
//!   * ZPIN   → `riscv_witness/checkpoint/zisk/zisk_program_input.h`
//!
//! These are the two blobs the riscv-witness ZisK perf-bench (#1917) consumes
//! via `--rom`/`--input`. Writers must produce byte-identical output. Field
//! order is normative — do not reorder.
use std::io::{self, Write};

use byteorder::{LittleEndian, WriteBytesExt};
use zisk_core::{ZiskInst, ZiskRom, ROM_ENTRY};

/// 'ZROM' little-endian.
const ZROM_MAGIC: u32 = 0x4D4F525A;
const ZROM_VERSION: u16 = 1;

/// 'ZPIN' little-endian. Program-input bundle: raw input bytes + RO sections
/// (read-only `.rodata`) + RW sections (writable `.data` initial image)
/// extracted from the ELF; the consumer splats both into emulator memory
/// before replay starts.
///
/// v2 (this writer) appends the RW-section block after the RO block; v1 had
/// only the RO block. The consumer reader accepts both — see
/// `ParseProgramInput` in `zisk_program_input.h`. The RW image is the v1.0.0-
/// alpha `ZiskRom::rw_data_64` that native loads via `Mem::init_write_section_data`;
/// without it, RAM `.data` starts all-zero and a real guest diverges early.
const ZPIN_MAGIC: u32 = 0x4E49505A;
const ZPIN_VERSION: u16 = 2;

/// Flattens upstream's three pc-keyed storage backings (entry / aligned /
/// non-aligned, see core/src/zisk_rom.rs:78-118) into a single ascending-paddr
/// slice. Upstream's own `sorted_pc_list` is the canonical iteration order;
/// each pc resolves through [`ZiskRom::get_instruction`].
pub fn collect_insts_in_pc_order(rom: &ZiskRom) -> Vec<&ZiskInst> {
    rom.sorted_pc_list.iter().map(|&pc| rom.get_instruction(pc)).collect()
}

/// Writes one `ZiskInst` in the 116-byte normative layout. Field order MUST
/// match `WriteInst` in zisk_rom_io.cpp:33-56 — do not reorder.
fn write_zisk_inst<W: Write>(w: &mut W, i: &ZiskInst) -> io::Result<()> {
    w.write_u64::<LittleEndian>(i.paddr)?;
    w.write_u8(u8::from(i.store_pc))?;
    w.write_u8(u8::from(i.store_use_sp))?;
    w.write_u64::<LittleEndian>(i.store)?;
    w.write_i64::<LittleEndian>(i.store_offset)?;
    w.write_u8(u8::from(i.set_pc))?;
    w.write_u8(u8::from(i.is_precompiled))?;
    w.write_u64::<LittleEndian>(i.ind_width)?;
    w.write_u8(u8::from(i.end))?;
    w.write_u64::<LittleEndian>(i.a_src)?;
    w.write_u64::<LittleEndian>(i.a_use_sp_imm1)?;
    w.write_u64::<LittleEndian>(i.a_offset_imm0)?;
    w.write_u64::<LittleEndian>(i.b_src)?;
    w.write_u64::<LittleEndian>(i.b_use_sp_imm1)?;
    w.write_u64::<LittleEndian>(i.b_offset_imm0)?;
    w.write_i64::<LittleEndian>(i.jmp_offset1)?;
    w.write_i64::<LittleEndian>(i.jmp_offset2)?;
    w.write_u8(u8::from(i.is_external_op))?;
    w.write_u8(i.op)?;
    // ZiskOperationType is #[repr(u32)] so the cast is a stable wire encoding.
    w.write_u32::<LittleEndian>(i.op_type as u32)?;
    w.write_u8(u8::from(i.m32))?;
    w.write_u64::<LittleEndian>(i.input_size)?;
    Ok(())
}

/// Serializes a transpiled `ZiskRom` to the canonical ZROM artifact. Header is
/// 24 bytes (magic + version + reserved + entry_pc + count), each instruction
/// is 116 bytes. Entry pc = `ROM_ENTRY` (0x1000): the native emulator always
/// boots from the BIOS entry, regardless of the ELF's `e_entry`.
pub fn write_zrom<W: Write>(w: &mut W, rom: &ZiskRom) -> io::Result<()> {
    let insts = collect_insts_in_pc_order(rom);
    // Fail fast if upstream ever stops booting at ROM_ENTRY — the wire field
    // is a *constant* on the producer side but a *variable* on the consumer
    // side, so any divergence would silently mis-trace.
    assert_eq!(
        rom.sorted_pc_list.first().copied(),
        Some(ROM_ENTRY),
        "ROM_ENTRY assumption violated — first paddr was {:?}",
        rom.sorted_pc_list.first(),
    );
    w.write_u32::<LittleEndian>(ZROM_MAGIC)?;
    w.write_u16::<LittleEndian>(ZROM_VERSION)?;
    w.write_u16::<LittleEndian>(0)?; // reserved
    w.write_u64::<LittleEndian>(ROM_ENTRY)?;
    w.write_u64::<LittleEndian>(insts.len() as u64)?;
    for i in &insts {
        write_zisk_inst(w, i)?;
    }
    Ok(())
}

/// Borrowed view of a ZPIN payload. `input` is the user-supplied per-execution
/// stdin bytes; `ro_sections` is `(start_addr, bytes)` from the ELF RO segments
/// (`ZiskRom::ro_data`); `rw_sections` is the writable `.data` initial image
/// (`ZiskRom::rw_data`).
pub struct ProgramInput<'a> {
    pub input: &'a [u8],
    pub ro_sections: &'a [(u64, &'a [u8])],
    pub rw_sections: &'a [(u64, &'a [u8])],
}

fn write_sections<W: Write>(w: &mut W, sections: &[(u64, &[u8])]) -> io::Result<()> {
    w.write_u64::<LittleEndian>(sections.len() as u64)?;
    for (start, data) in sections {
        w.write_u64::<LittleEndian>(*start)?;
        w.write_u64::<LittleEndian>(data.len() as u64)?;
        w.write_all(data)?;
    }
    Ok(())
}

/// Serializes a ZPIN bundle (v2). Layout (LE):
///   header  : magic u32 + version u16 + reserved u16
///   input   : len u64 + bytes
///   ro      : count u64 + repeat { start u64 + len u64 + bytes }
///   rw      : count u64 + repeat { start u64 + len u64 + bytes }
pub fn write_zpin<W: Write>(w: &mut W, p: &ProgramInput) -> io::Result<()> {
    w.write_u32::<LittleEndian>(ZPIN_MAGIC)?;
    w.write_u16::<LittleEndian>(ZPIN_VERSION)?;
    w.write_u16::<LittleEndian>(0)?;
    w.write_u64::<LittleEndian>(p.input.len() as u64)?;
    w.write_all(p.input)?;
    write_sections(w, p.ro_sections)?;
    write_sections(w, p.rw_sections)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::ReadBytesExt;
    use std::io::Cursor;
    use zisk_core::ZiskOperationType;

    /// Per-instruction wire stride: 13*u64 + 7*u8 + 1*u8(op) + 1*u32 = 116.
    const INST_BYTES: usize = 13 * 8 + 7 * 1 + 1 + 4;
    /// Header: u32 + u16 + u16 + u64 + u64 = 24.
    const HEADER_BYTES: usize = 24;

    fn sample_inst(paddr: u64) -> ZiskInst {
        let mut i = ZiskInst::default();
        i.paddr = paddr;
        i.store_pc = true;
        i.store = 3;
        i.store_offset = -7;
        i.set_pc = true;
        i.ind_width = 4;
        i.end = true;
        i.a_src = 6;
        i.a_offset_imm0 = 0x1234_5678_9ABC_DEF0;
        i.b_src = 2;
        i.b_offset_imm0 = 0xDEAD_BEEF;
        i.jmp_offset1 = -42;
        i.jmp_offset2 = 99;
        i.is_external_op = true;
        i.op = 0x2c;
        i.op_type = ZiskOperationType::Binary;
        i.m32 = true;
        i.input_size = 32;
        i
    }

    /// Bytes-out re-parsed and re-emitted must match — proves the writer is
    /// deterministic and field-order-stable. (C++ side covers full round-trip
    /// against the canonical reader in zisk_rom_test.cpp.)
    #[test]
    fn write_zisk_inst_round_trips() {
        let inst = sample_inst(0x80000000);
        let mut buf = Vec::new();
        write_zisk_inst(&mut buf, &inst).unwrap();
        assert_eq!(buf.len(), INST_BYTES);

        // Read each field back in the same order; re-emit; compare bytes.
        let mut r = Cursor::new(&buf);
        let mut out = Vec::new();
        macro_rules! pass_u64 { () => { out.write_u64::<LittleEndian>(r.read_u64::<LittleEndian>().unwrap()).unwrap(); }; }
        macro_rules! pass_i64 { () => { out.write_i64::<LittleEndian>(r.read_i64::<LittleEndian>().unwrap()).unwrap(); }; }
        macro_rules! pass_u8  { () => { out.write_u8(r.read_u8().unwrap()).unwrap(); }; }
        pass_u64!();                                                              // paddr
        pass_u8!(); pass_u8!();                                                   // store_pc, store_use_sp
        pass_u64!(); pass_i64!();                                                 // store, store_offset
        pass_u8!(); pass_u8!();                                                   // set_pc, is_precompiled
        pass_u64!();                                                              // ind_width
        pass_u8!();                                                               // end
        pass_u64!(); pass_u64!(); pass_u64!();                                    // a_src, a_use_sp_imm1, a_offset_imm0
        pass_u64!(); pass_u64!(); pass_u64!();                                    // b_src, b_use_sp_imm1, b_offset_imm0
        pass_i64!(); pass_i64!();                                                 // jmp_offset1, jmp_offset2
        pass_u8!(); pass_u8!();                                                   // is_external_op, op
        out.write_u32::<LittleEndian>(r.read_u32::<LittleEndian>().unwrap()).unwrap(); // op_type
        pass_u8!();                                                               // m32
        pass_u64!();                                                              // input_size
        assert_eq!(buf, out);
    }

    /// Header bytes + count + stride. Doesn't require a real transpiled ROM —
    /// fabricates a one-inst ZiskRom by populating `sorted_pc_list` +
    /// `rom_entry_instructions` directly. `min_program_pc` is set above
    /// `ROM_ENTRY` so `get_instruction(ROM_ENTRY)` falls into the entry branch
    /// (see core/src/zisk_rom.rs:127-165).
    #[test]
    fn write_zrom_header_and_stride() {
        let mut rom = ZiskRom::default();
        rom.min_program_pc = 0x8000_0000;
        let inst = sample_inst(ROM_ENTRY);
        rom.rom_bios_instructions.push(inst);
        rom.sorted_pc_list.push(ROM_ENTRY);

        let mut buf = Vec::new();
        write_zrom(&mut buf, &rom).unwrap();
        assert_eq!(buf.len(), HEADER_BYTES + INST_BYTES);

        // magic = 'ZROM' LE
        assert_eq!(&buf[0..4], &[0x5A, 0x52, 0x4F, 0x4D]);
        // version = 1, reserved = 0
        assert_eq!(&buf[4..8], &[0x01, 0x00, 0x00, 0x00]);
        // entry_pc = ROM_ENTRY (0x1000)
        assert_eq!(u64::from_le_bytes(buf[8..16].try_into().unwrap()), ROM_ENTRY);
        // inst count = 1
        assert_eq!(u64::from_le_bytes(buf[16..24].try_into().unwrap()), 1);
    }

    /// ZPIN header + lengths + section stride. No `ZiskRom` needed.
    #[test]
    fn write_zpin_header_and_layout() {
        let ro_a: Vec<u8> = (0..16).collect();
        let ro_b: Vec<u8> = vec![0xAB; 4];
        let sections = [(0xA0000000u64, ro_a.as_slice()), (0xB0000000u64, ro_b.as_slice())];
        let rw_c: Vec<u8> = vec![0xCD; 8];
        let rw_sections = [(0xA0020000u64, rw_c.as_slice())];
        let input: Vec<u8> = (0..8).collect();
        let p = ProgramInput { input: &input, ro_sections: &sections, rw_sections: &rw_sections };

        let mut buf = Vec::new();
        write_zpin(&mut buf, &p).unwrap();

        // Layout: 8 hdr + 8 input_len + 8 input + 8 ro_count
        //       + per-ro-section (8 start + 8 len + bytes)
        //       + 8 rw_count + per-rw-section (8 start + 8 len + bytes)
        let expected = 8 + 8 + input.len() + 8 + (16 + ro_a.len()) + (16 + ro_b.len())
            + 8 + (16 + rw_c.len());
        assert_eq!(buf.len(), expected);

        // magic = 'ZPIN' LE
        assert_eq!(&buf[0..4], &[0x5A, 0x50, 0x49, 0x4E]);
        // version = 2, reserved
        assert_eq!(&buf[4..8], &[0x02, 0x00, 0x00, 0x00]);
        // input_len
        assert_eq!(u64::from_le_bytes(buf[8..16].try_into().unwrap()), input.len() as u64);
        assert_eq!(&buf[16..16 + input.len()], &input[..]);

        let mut p_off = 16 + input.len();
        assert_eq!(u64::from_le_bytes(buf[p_off..p_off + 8].try_into().unwrap()), 2);
        p_off += 8;
        assert_eq!(u64::from_le_bytes(buf[p_off..p_off + 8].try_into().unwrap()), 0xA0000000);
        p_off += 8;
        assert_eq!(u64::from_le_bytes(buf[p_off..p_off + 8].try_into().unwrap()), ro_a.len() as u64);
        p_off += 8;
        assert_eq!(&buf[p_off..p_off + ro_a.len()], &ro_a[..]);
        p_off += ro_a.len();
        assert_eq!(u64::from_le_bytes(buf[p_off..p_off + 8].try_into().unwrap()), 0xB0000000);
    }

    /// Empty ZPIN (no input, no sections) is 40 bytes — header + input_len +
    /// two zero-length section counts (ro + rw). Catches accidental writes that
    /// depend on `input.len() > 0`.
    #[test]
    fn write_zpin_empty() {
        let p = ProgramInput { input: &[], ro_sections: &[], rw_sections: &[] };
        let mut buf = Vec::new();
        write_zpin(&mut buf, &p).unwrap();
        assert_eq!(buf.len(), 8 + 8 + 8 + 8);
        assert_eq!(&buf[0..4], &[0x5A, 0x50, 0x49, 0x4E]);
        assert_eq!(u64::from_le_bytes(buf[8..16].try_into().unwrap()), 0); // input_len
        assert_eq!(u64::from_le_bytes(buf[16..24].try_into().unwrap()), 0); // ro_count
        assert_eq!(u64::from_le_bytes(buf[24..32].try_into().unwrap()), 0); // rw_count
    }
}

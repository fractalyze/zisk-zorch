//! rw fixture-gen — local PoC tool (NOT upstreamed; lives in fractalyze/zisk).
//!
//! Drives native ZisK state machines to produce byte-parity fixtures for
//! riscv-witness. PR-Z1:
//!   T-Z1.1 — stand up the proofman setup context (`ProofCtx`+`SetupCtx`+`Std`)
//!            and construct `BinaryBasicSM` (`--selftest`).
//!   T-Z1.2 — emit an `add_single` Binary fixture: drive `compute_witness`,
//!            extract the trace as canonical u64, write a gzipped `.npy` plus
//!            `input_records.bin` + `fixture_metadata.json`.
//!
//! Setup recipe mirrors `cli/src/commands/program_setup.rs` + `executor/src/utils.rs`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

mod zisk_wire;
use zisk_wire::ProgramInput;

use anyhow::{bail, Context, Result};
use clap::Parser;
use fields::{Goldilocks, PrimeField64};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use pil_std_lib::Std;
use proofman::ProofMan;
use proofman_common::{
    init_gpu_setup, AirInstance, MpiCtx, ProofCtx, ProofType, SetupCtx, SetupsVadcop, TraceInfo,
    VerboseMode,
};
use sha2::{Digest, Sha256};
use sm_binary::{
    BinaryBasicSM, BinaryExtensionSM, BinaryInput, ADDW_OP, ADD_OP, AND_OP, EQW_OP, EQ_OP, GT_OP,
    LEUW_OP, LEU_OP, LEW_OP, LE_OP, LTUW_OP, LTU_OP, LTW_OP, LT_ABS_NP_OP, LT_ABS_PN_OP, LT_OP,
    MAXUW_OP, MAXU_OP, MAXW_OP, MAX_OP, MINUW_OP, MINU_OP, MINW_OP, MIN_OP, OR_OP, SEXT_B_OP,
    SEXT_H_OP, SEXT_W_OP, SLLW_OP, SLL_OP, SRAW_OP, SRA_OP, SRLW_OP, SRL_OP, SUBW_OP, SUB_OP,
    XOR_OP,
};
use precomp_arith_eq::{
    Arith256Input, Arith256ModInput, ArithEqInput, ArithEqSM, Bn254ComplexAddInput,
    Bn254ComplexMulInput, Bn254ComplexSubInput, Bn254CurveAddInput, Bn254CurveDblInput,
    Secp256k1AddInput, Secp256k1DblInput, Secp256r1AddInput, Secp256r1DblInput,
};
use precomp_arith_eq_384::{
    Arith384ModInput, ArithEq384Input, ArithEq384SM, Bls12_381ComplexAddInput,
    Bls12_381ComplexMulInput, Bls12_381ComplexSubInput, Bls12_381CurveAddInput,
    Bls12_381CurveDblInput,
};
use precomp_keccakf::{KeccakfInput, KeccakfSM};
use precomp_sha256f::{Sha256fInput, Sha256fSM};
use zisk_common::{
    BusId, CollectSkipper, ExtOperationData, OperationData, ZiskPaths, A, B, OP, OPERATION_BUS_ID,
};
use sm_binary::{
    BinaryAddCollector, BinaryAddInstance, BinaryAddSM, BinaryBasicCollector, BinaryBasicInstance,
    BinaryCounter, BinaryExtensionCollector, BinaryExtensionInstance, BinarySM,
};
use sm_arith::{ArithFrops, ArithFullSM};
use sm_main::{MainInstance, MainPlanner};
use sm_mem::{
    MemAlignByteInstance, MemAlignByteSM, MemAlignInput, MemAlignInstance,
    MemAlignReadByteInstance, MemAlignSM,
    MemAlignWriteByteInstance, MemInput, MemModule, MemModuleInstance, MemPlanner,
    MemPreviousSegment, MemSM, InputDataSM, RomDataSM,
};
use mem_common::MemCounters;
use zisk_common::{
    plan as plan_instance_windows, BusDevice, BusDeviceMetrics, CheckPoint, ChunkId,
    ComponentPlanBuilder, EmuTrace, InstCount, Instance, InstanceCtx, InstanceType, Planner,
    MEM_BUS_ID,
};
use data_bus::DataBusTrait;
use zisk_core::{Riscv2zisk, ZiskRom};
use ziskemu::{EmuOptions, ZiskEmulator};
use zisk_pil::{
    ArithEq384Trace, ArithEq384TraceRow, ArithEqTrace, ArithEqTraceRow, ArithTrace, ArithTraceRow,
    BinaryAddTrace, BinaryAddTraceRow, BinaryExtensionTrace,
    BinaryExtensionTraceRow, BinaryTrace, BinaryTraceRow, KeccakfTrace, KeccakfTraceRow,
    MemAlignByteTrace, MemAlignByteTraceRow,
    MemAlignReadByteTrace, MemAlignReadByteTraceRow, MemAlignTrace, MemAlignTraceRow,
    MemAlignWriteByteTrace, MemAlignWriteByteTraceRow, MemTrace, MemTraceRow, InputDataTrace,
    InputDataTraceRow, MainTrace, MainTraceRow, RomDataTrace, RomDataTraceRow, Sha256fTrace,
    Sha256fTraceRow, MEM_AIR_IDS, ZISK_AIRGROUP_ID,
    MEM_ALIGN_AIR_IDS, MEM_ALIGN_BYTE_AIR_IDS, MEM_ALIGN_READ_BYTE_AIR_IDS, MEM_ALIGN_WRITE_BYTE_AIR_IDS,
    INPUT_DATA_AIR_IDS, ROM_DATA_AIR_IDS,
};

type F = Goldilocks;

#[derive(Parser, Debug)]
#[command(about = "rw fixture-gen (PoC): drive native ZisK SMs to emit parity fixtures")]
struct Args {
    /// Only stand up the setup context + construct the SM, then exit (T-Z1.1).
    #[arg(long)]
    selftest: bool,
    /// Chip to generate a fixture for (e.g. "binary").
    #[arg(long)]
    chip: Option<String>,
    /// Case name (e.g. "add_single").
    #[arg(long)]
    case: Option<String>,
    /// Output directory for the fixture.
    #[arg(long)]
    out: Option<String>,
    /// Also write the full native trace (gzipped .npy, row-major canonical u64
    /// LE — the exact `golden_sha256` preimage) next to each fixture, on the
    /// single-opcode AND full-program paths. Not committed — the committed
    /// fixture stays input_records.bin + metadata.json (golden hash only); the
    /// blob feeds downstream regeneration recipes (e.g. zisk-zorch real-trace
    /// stage-1 fixtures) and local first-diff hunting.
    #[arg(long)]
    debug_trace: bool,
    /// Stage-1 commit oracle (zisk-zorch #83): point at a fixture dir holding a
    /// `--debug-trace` dump (`expected_<chip>_trace.npy.gz` +
    /// `fixture_metadata.json`), reload the trace into an `AirInstance`, and run
    /// the native stage-1 commit on it (`commit_witness` = witness-expr hints +
    /// coset LDE + MerkleTreeGL) exactly as a real prove would. Writes
    /// `stage1_commit.json` into the same dir: the 4-element root plus the
    /// starkStruct params a downstream byte-match needs to reproduce it.
    #[arg(long)]
    stage1_root: Option<String>,
    /// Tree/leaf hash family for `--stage1-root`, overriding the proving key's
    /// globalInfo (whose absent `hash` field means proofman's default,
    /// currently Poseidon1). Lets a downstream that models the other family
    /// (zisk-zorch commits with Poseidon2) capture a matching root without
    /// editing the installed key. Recorded as `hash_family` in the output.
    #[arg(long)]
    hash_family: Option<String>,
    /// Full-program mode (Gate-0): a guest ELF to transpile + emulate, collecting
    /// a real run's per-chip inputs off the operation bus instead of synthetic ones.
    #[arg(long)]
    elf: Option<String>,
    /// Optional guest stdin byte stream for `--elf`. Absent ⇒ empty input.
    #[arg(long)]
    inputs: Option<String>,
    /// Mem spike (#1845): with `--elf`, run the MemCounters count pass + MemPlanner
    /// and report the native segmentation (segments per mem module) instead of
    /// emitting fixtures. De-risks the segmented-mem oracle design.
    #[arg(long)]
    mem_spike: bool,
    /// Emit the Mem (air_id 14) full-program fixture (#1845): drive the count→plan
    /// → per-chunk-collector → compute_witness pipeline for the Mem module.
    #[arg(long)]
    mem: bool,
    /// Emit the MemAlign (air_id 5) full-program fixture (#1845): same pipeline
    /// for the unaligned-access sub-machine (no segment carry).
    #[arg(long)]
    mem_align: bool,
    /// Emit the MemAlignByte (air_id 6) full-program fixtures (#1845):
    /// per-segment byte-granular unaligned accesses. Multi-segment, no carry.
    #[arg(long)]
    mem_align_byte: bool,
    /// Emit the MemAlignReadByte (air_id 7) full-program fixtures (#1845):
    /// per-segment byte-granular unaligned reads. Multi-segment, no carry.
    #[arg(long)]
    mem_align_read_byte: bool,
    /// Emit the MemAlignWriteByte (air_id 8) full-program fixtures (#1845):
    /// per-segment byte-granular unaligned writes. Multi-segment, no carry.
    #[arg(long)]
    mem_align_write_byte: bool,
    /// Emit the binary-family full-program fixtures plan-driven and
    /// multi-instance (#2347): count→plan→expand via the native BinaryPlanner
    /// (including its enable_bin_add_sm cost split), one `instNN/` fixture per
    /// planned BinaryBasic / BinaryAdd / BinaryExtension instance. Unlike the
    /// bare `--elf` Gate-0 path, this handles runs whose op counts span
    /// multiple AIR instances (e.g. an eth block).
    #[arg(long)]
    binary_multi: bool,
    /// Emit the Keccakf full-program fixtures plan-driven and multi-instance
    /// (#2347): the native (skip, count) windows over the chunk-ordered perm
    /// stream, one `instNN/` fixture per planned instance. Unlike the bare
    /// `--elf` Gate-0 path, this handles runs whose perm count exceeds one
    /// 131072-row instance (e.g. an eth block).
    #[arg(long)]
    keccak_multi: bool,
    /// Emit the Main-SM per-segment goldens (#2347 follow-on, the milestone's
    /// main-SM gate): native MainPlanner segmentation + MainInstance witness
    /// per segment, metadata-only fixtures (the rw gate builds its own step
    /// stream from the committed ZROM/ZPIN).
    #[arg(long)]
    main_multi: bool,
    /// Emit the RomData (air_id 15) full-program fixture (#1887): same
    /// count→plan→expand pipeline as `--mem` for the read-only ROM-data module.
    #[arg(long)]
    rom_data: bool,
    /// Emit the InputData (air_id 16) full-program fixture (#1887): same
    /// count→plan→expand pipeline as `--rom-data` for the input-data module.
    #[arg(long)]
    input_data: bool,
    /// Dump the ZROM (`--rom-out`) + ZPIN (`--program-input-out`) blobs the
    /// riscv-witness ZisK perf-bench (#1917) consumes via its `--rom`/`--input`
    /// flags. Pure transpile of `--elf` (+ raw `--inputs` bytes) — no emulator,
    /// no proving key, so it runs proofman-free and skips `build_std`. The
    /// perf-bench regenerates checkpoints itself, so no checkpoint blob is
    /// emitted here.
    #[arg(long)]
    emu_trace_dump: bool,
    /// ZROM output path for `--emu-trace-dump`.
    #[arg(long)]
    rom_out: Option<String>,
    /// ZPIN output path for `--emu-trace-dump`.
    #[arg(long)]
    program_input_out: Option<String>,
    /// Provenance JSON output path for `--emu-trace-dump`.
    #[arg(long)]
    metadata_out: Option<String>,
    /// Producer fork commit recorded as `zisk_commit` in the dump metadata.
    /// Supply it explicitly when the cwd is not this fork's worktree (e.g. the
    /// Bazel genrule runs in the consumer's execroot, where a `git rev-parse`
    /// would resolve the wrong repo). Omit it for a standalone run from the fork
    /// worktree, where it falls back to that worktree's HEAD.
    #[arg(long)]
    zisk_commit: Option<String>,
    /// Emit a per-chunk execute-seed reference JSON to this path (#2067): runs
    /// the emulator over `--elf` (+ optional `--inputs`) and writes, per
    /// checkpoint chunk, the start pc / clk / 32-register seed in the
    /// zkVM-agnostic schema riscv-witness' `checkpoint_seed_parity.h` parses.
    /// Proofman-free (transpile + emulate, no proving key), like
    /// `--emu-trace-dump`.
    #[arg(long)]
    checkpoint_reference: Option<String>,
    /// Per-chunk step budget for `--checkpoint-reference` (= the rw consumer's
    /// `CheckpointConfig::checkpoint_size`). Seed boundaries must match the rw
    /// checkpoint stream's, so this mirrors the value the consumer test passes.
    /// Defaults to 2^18 (the emulator's default chunk size).
    #[arg(long)]
    chunk_size: Option<u64>,
}

/// Feeds one chunk's `MEM_BUS_ID` writes to a single `MemAlignCollector`.
struct MemAlignFeedBus {
    collector: sm_mem::MemAlignCollector,
}

impl DataBusTrait<u64, ()> for MemAlignFeedBus {
    fn write_to_bus(&mut self, bus_id: BusId, data: &[u64], _data_ext: &[u64]) -> bool {
        if bus_id == MEM_BUS_ID {
            self.collector.process_data(&bus_id, data);
        }
        true
    }

    fn on_close(&mut self) {}

    fn into_devices(self, _execute_on_close: bool) -> Vec<(usize, ())> {
        Vec::new()
    }
}

/// Feeds one chunk's `OPERATION_BUS_ID` ops to a per-chunk `BinaryCounter`
/// (the count phase of the plan-driven binary emission, #2347).
struct BinaryCountBus {
    counter: BinaryCounter,
}

impl DataBusTrait<u64, ()> for BinaryCountBus {
    fn write_to_bus(&mut self, bus_id: BusId, data: &[u64], _data_ext: &[u64]) -> bool {
        if bus_id == OPERATION_BUS_ID {
            self.counter.process_data(&bus_id, data);
        }
        true
    }

    fn on_close(&mut self) {}

    fn into_devices(self, _execute_on_close: bool) -> Vec<(usize, ())> {
        Vec::new()
    }
}

/// Feeds one chunk's `OPERATION_BUS_ID` ops to a plan-built binary-family
/// collector during the expand pass (#2347). The three binary collectors share
/// the `process_data(&BusId, &[u64])` shape but are distinct concrete types
/// (`BusDevice` is a marker trait), so the feed bus is macro-stamped per type.
macro_rules! binary_feed_bus {
    ($name:ident, $collector:ty) => {
        struct $name {
            collector: $collector,
        }

        impl DataBusTrait<u64, ()> for $name {
            fn write_to_bus(&mut self, bus_id: BusId, data: &[u64], _data_ext: &[u64]) -> bool {
                if bus_id == OPERATION_BUS_ID {
                    self.collector.process_data(&bus_id, data);
                }
                true
            }

            fn on_close(&mut self) {}

            fn into_devices(self, _execute_on_close: bool) -> Vec<(usize, ())> {
                Vec::new()
            }
        }
    };
}

binary_feed_bus!(BinaryBasicFeedBus, BinaryBasicCollector<F>);
binary_feed_bus!(BinaryAddFeedBus, BinaryAddCollector<F>);
binary_feed_bus!(BinaryExtensionFeedBus, BinaryExtensionCollector<F>);

/// Captures one chunk's full Keccakf op stream, in bus order. The plan-driven
/// multi-instance keccak emission (#2347) slices these per-chunk streams by the
/// planner's (skip, count) windows instead of driving the macro-generated
/// `KeccakfCollector` (whose `inputs` field is crate-private) — the slice
/// semantics are identical: skip the first `skip` matching ops, take `count`.
struct KeccakCaptureBus {
    inputs: Vec<KeccakfInput>,
}

impl DataBusTrait<u64, ()> for KeccakCaptureBus {
    fn write_to_bus(&mut self, bus_id: BusId, data: &[u64], _data_ext: &[u64]) -> bool {
        if bus_id == OPERATION_BUS_ID && data[1] == zisk_core::ZiskOperationType::Keccak as u64 {
            match data.try_into() {
                Ok(ExtOperationData::OperationKeccakData(d)) => {
                    self.inputs.push(KeccakfInput::from(&d));
                }
                _ => {
                    panic!("Expected ExtOperationData::OperationKeccakData for Keccak operation")
                }
            }
        }
        true
    }

    fn on_close(&mut self) {}

    fn into_devices(self, _execute_on_close: bool) -> Vec<(usize, ())> {
        Vec::new()
    }
}

/// Feeds one chunk's `MEM_BUS_ID` writes to a single `MemModuleCollector` (built
/// from a segment plan) during the expand pass.
struct MemFeedBus {
    collector: sm_mem::MemModuleCollector,
}

impl DataBusTrait<u64, ()> for MemFeedBus {
    fn write_to_bus(&mut self, bus_id: BusId, data: &[u64], _data_ext: &[u64]) -> bool {
        if bus_id == MEM_BUS_ID {
            self.collector.process_data(&bus_id, data);
        }
        true
    }

    fn on_close(&mut self) {}

    fn into_devices(self, _execute_on_close: bool) -> Vec<(usize, ())> {
        Vec::new()
    }
}

/// Minimal `DataBusTrait` for the mem count pass: routes `MEM_BUS_ID` writes to a
/// per-chunk `MemCounters` (which the `MemPlanner` later turns into segment plans).
struct MemCountBus {
    counters: MemCounters,
    mem_writes_seen: u64,
}

impl DataBusTrait<u64, ()> for MemCountBus {
    fn write_to_bus(&mut self, bus_id: BusId, data: &[u64], _data_ext: &[u64]) -> bool {
        if bus_id == MEM_BUS_ID {
            self.mem_writes_seen += 1;
            self.counters.process_data(&bus_id, data);
        }
        true
    }

    fn on_close(&mut self) {}

    fn into_devices(self, _execute_on_close: bool) -> Vec<(usize, ())> {
        Vec::new()
    }
}

/// Spike: drive the prover's mem count→plan phases over a guest run and report the
/// segmentation, to verify the planner is drivable outside the orchestrator and to
/// size the segmented-mem oracle (#1845).
/// Canonical-u64 SHA-256 over a trace's field elements — the fixture golden.
/// Hashes each row-major element's little-endian canonical u64 on the fly, so
/// it never materializes the whole trace as an intermediate `Vec<u64>`.
fn trace_golden_sha256(trace: &[F]) -> String {
    let mut hasher = Sha256::new();
    for f in trace {
        hasher.update(f.as_canonical_u64().to_le_bytes());
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// `--debug-trace` dump for the field-element emit paths: the trace as a gzipped
/// `.npy` whose payload is exactly the `golden_sha256` preimage (row-major
/// canonical u64, little-endian). Materializes the canonical `Vec<u64>` — only
/// runs under the flag, so the hash-only path keeps its streaming behavior.
fn write_trace_dump(path: &Path, trace: &[F], rows: usize, n_cols: usize) -> Result<()> {
    let data: Vec<u64> = trace.iter().map(|f| f.as_canonical_u64()).collect();
    write_npy_gz(path, &data, rows, n_cols)
}

/// Transpile a guest ELF and run the emulator to its per-chunk minimal traces —
/// the shared front half of every full-program oracle path. Uses the emulator's
/// default 2^18 chunk size; the seed oracle, whose chunk boundaries must match
/// the rw checkpoint stream, calls `load_guest_min_traces_with_chunk_size`.
fn load_guest_min_traces(
    elf_path: &str,
    inputs_path: Option<&str>,
) -> Result<(ZiskRom, Vec<EmuTrace>)> {
    load_guest_min_traces_with_chunk_size(elf_path, inputs_path, 1 << 18)
}

/// The plan's chunks in execution order.
///
/// The planner emits them unordered. Chunk id *is* execution order, and the mem
/// state machines difference steps within an address, so collectors fed in the
/// planner's order deliver an address's ops out of step order and the
/// difference underflows (fractalyze/riscv-witness#2325).
fn plan_chunk_ids(check_point: &CheckPoint, air_label: &str) -> Result<Vec<ChunkId>> {
    let mut chunk_ids = match check_point {
        CheckPoint::Multiple(v) => v.clone(),
        CheckPoint::Single(c) => vec![*c],
        _ => bail!("unexpected {air_label} checkpoint variant"),
    };
    chunk_ids.sort_unstable();
    Ok(chunk_ids)
}

/// `load_guest_min_traces` with a caller-chosen chunk size. The seed-reference
/// oracle (#2067) needs the chunk size to equal the rw consumer's
/// `CheckpointConfig::checkpoint_size`, so each minimal trace lines up 1:1 with
/// an rw checkpoint.
fn load_guest_min_traces_with_chunk_size(
    elf_path: &str,
    inputs_path: Option<&str>,
    chunk_size: u64,
) -> Result<(ZiskRom, Vec<EmuTrace>)> {
    let elf_bytes = fs::read(elf_path)?;
    let rom: ZiskRom = Riscv2zisk::new(&elf_bytes)
        .run()
        .map_err(|e| anyhow::anyhow!("riscv2zisk transpile failed: {e}"))?;
    let input_data: Vec<u8> = match inputs_path {
        Some(p) => fs::read(p)?,
        None => Vec::new(),
    };
    let emu_options = EmuOptions {
        chunk_size: Some(chunk_size),
        max_steps: 0xF_FFFF_FFFF,
        ..EmuOptions::default()
    };
    let min_traces = ZiskEmulator::compute_minimal_traces(&rom, &input_data, &emu_options, 1)?;
    Ok((rom, min_traces))
}

fn mem_spike(elf_path: &str, inputs_path: Option<&str>) -> Result<()> {
    let (rom, min_traces) = load_guest_min_traces(elf_path, inputs_path)?;

    // Count phase: one MemCounters per chunk, fed MEM_BUS_ID via with_mem_ops=true.
    let mut metrics: Vec<(ChunkId, Box<dyn BusDeviceMetrics>)> = Vec::new();
    let mut total_mem_writes = 0u64;
    for (i, emu_trace) in min_traces.iter().enumerate() {
        let mut bus = MemCountBus { counters: MemCounters::new(), mem_writes_seen: 0 };
        // The executor seeds the FIRST chunk's counter with the memory-init
        // sections before busing it (execution/rust.rs, `is_first()`); the
        // planner's offsets under-allocate every initialized address without
        // this, and the trace rows for the init writes land as overwrites.
        if i == 0 {
            bus.counters.init_with_mem_sections(&rom as &dyn zisk_core::MemDataSection);
        }
        ZiskEmulator::process_emu_trace::<F, (), MemCountBus>(&rom, emu_trace, &mut bus, true);
        total_mem_writes += bus.mem_writes_seen;
        // close() builds the per-module addr_sorted partitions the Mem/Rom/Input
        // planners read; the executor triggers it via into_devices(true).
        bus.counters.close();
        metrics.push((ChunkId(i), Box::new(bus.counters)));
    }
    println!("mem spike: {total_mem_writes} raw MEM_BUS_ID writes seen across all chunks");

    // Plan phase: the planner cuts the run into per-module, per-segment plans.
    let plans = MemPlanner::new().plan(metrics);
    let mut by_air: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for p in &plans {
        *by_air.entry(p.air_id).or_insert(0) += 1;
    }
    println!(
        "mem spike: {} chunks → {} plans (segments) across {} module AIR(s)",
        min_traces.len(),
        plans.len(),
        by_air.len(),
    );
    for (air_id, segs) in &by_air {
        println!("  air_id {air_id}: {segs} segment(s)");
    }
    for (i, p) in plans.iter().enumerate() {
        let chunks = match &p.check_point {
            CheckPoint::Single(c) => format!("Single({})", c.0),
            CheckPoint::Multiple(v) => format!("Multiple({} chunks)", v.len()),
            _ => "other".to_string(),
        };
        println!(
            "  plan[{i}]: airgroup={} air_id={} segment_id={:?} check_point={chunks}",
            p.airgroup_id, p.air_id, p.segment_id,
        );
    }
    Ok(())
}

/// Slice-1 of the segmented-mem oracle (#1845): emit the Mem (air_id 14)
/// full-program fixture by driving the prover's count→plan→expand pipeline. Only
/// the Mem module; assumes a single segment (true for guests whose RAM footprint
/// fits one AIR instance, e.g. go_hello_world — see `--mem-spike`).
#[allow(clippy::too_many_arguments)]
fn emit_mem(
    elf_path: &str,
    inputs_path: Option<&str>,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
    pctx: Arc<ProofCtx<F>>,
) -> Result<()> {
    let (rom, min_traces) = load_guest_min_traces(elf_path, inputs_path)?;

    // Count → plan (close() builds the per-module addr partitions; see --mem-spike).
    let mut metrics: Vec<(ChunkId, Box<dyn BusDeviceMetrics>)> = Vec::new();
    for (i, emu_trace) in min_traces.iter().enumerate() {
        let mut bus = MemCountBus { counters: MemCounters::new(), mem_writes_seen: 0 };
        // The executor seeds the FIRST chunk's counter with the memory-init
        // sections before busing it (execution/rust.rs, `is_first()`); the
        // planner's offsets under-allocate every initialized address without
        // this, and the trace rows for the init writes land as overwrites.
        if i == 0 {
            bus.counters.init_with_mem_sections(&rom as &dyn zisk_core::MemDataSection);
        }
        ZiskEmulator::process_emu_trace::<F, (), MemCountBus>(&rom, emu_trace, &mut bus, true);
        bus.counters.close();
        metrics.push((ChunkId(i), Box::new(bus.counters)));
    }
    let mut mem_plans: Vec<_> = MemPlanner::new()
        .plan(metrics)
        .into_iter()
        .filter(|p| p.air_id == MEM_AIR_IDS[0])
        .collect();
    if mem_plans.is_empty() {
        bail!("planner produced no Mem (air_id 14) plan");
    }
    mem_plans.sort_by_key(|p| p.segment_id.map(|s| s.0).unwrap_or(0));
    let total_segments = mem_plans.len();
    // A Mem segment fixture is large (≤ NUM_ROWS ops); seg 0 (prev = RAM base) plus
    // one carried segment proves the multi-segment carry without 12× the bytes.
    const MAX_MEM_SEGMENTS: usize = 2;

    let mem_module = MemSM::new(std.clone());
    let (min_addr, _max_addr) = mem_module.get_addr_range();
    for plan in mem_plans.into_iter().take(MAX_MEM_SEGMENTS) {
        let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
        emit_one_mem_segment(
            &rom,
            &min_traces,
            plan,
            &out.join(format!("seg{seg:02}")),
            elf_path,
            debug_trace,
            min_addr,
            total_segments,
            std.clone(),
            &sctx,
            &pctx,
        )?;
    }
    Ok(())
}

/// Expand + emit one Mem segment: build a collector per chunk in the segment's
/// plan, feed it, extract the inputs + previous-segment carry for the fixture, and
/// drive `MemModuleInstance::compute_witness` for the golden. The carry is what
/// makes this segmented — segment 0's prev is the RAM base, later segments inherit
/// the previous segment's last (addr, step, value) from their first collector.
#[allow(clippy::too_many_arguments)]
fn emit_one_mem_segment(
    rom: &ZiskRom,
    min_traces: &[zisk_common::EmuTrace],
    plan: zisk_common::Plan,
    out: &Path,
    elf_path: &str,
    debug_trace: bool,
    min_addr: u32,
    total_segments: usize,
    std: Arc<Std<F>>,
    sctx: &Arc<SetupCtx<F>>,
    pctx: &Arc<ProofCtx<F>>,
) -> Result<()> {
    let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
    let chunk_ids = plan_chunk_ids(&plan.check_point, "mem")?;
    let instance = MemModuleInstance::new(MemSM::new(std), InstanceCtx::new(0, plan));
    let mut collectors: Vec<(usize, Box<dyn BusDevice<u64>>)> = Vec::new();
    let mut serial: Vec<MemInput> = Vec::new();
    let mut prev = MemPreviousSegment { addr: min_addr, step: 0, value: 0 };
    for cid in &chunk_ids {
        let collector = instance.build_mem_collector(*cid, rom);
        let mut bus = MemFeedBus { collector };
        ZiskEmulator::process_emu_trace::<F, (), MemFeedBus>(rom, &min_traces[cid.0], &mut bus, true);
        let collector = bus.collector;
        for mi in &collector.inputs {
            serial.push(MemInput::new(mi.addr, mi.is_write, mi.step, mi.value));
        }
        if let Some(ps) = &collector.prev_segment {
            prev = MemPreviousSegment { addr: ps.addr, step: ps.step, value: ps.value };
        }
        collectors.push((cid.0, Box::new(collector) as Box<dyn BusDevice<u64>>));
    }

    let num_rows = MemTrace::<MemTraceRow<F>>::NUM_ROWS;
    let buffer = vec![F::default(); num_rows * MemTrace::<MemTraceRow<F>>::ROW_SIZE];
    let air = instance
        .compute_witness(pctx, sctx, collectors, buffer, false)?
        .ok_or_else(|| anyhow::anyhow!("Mem instance produced no AirInstance"))?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let golden_sha256 = trace_golden_sha256(&air.trace);

    // rw's BuildMemInputs consumes ops in (addr, step) order; mirror native
    // MemModuleInstance::prepare_inputs so the records match the golden trace.
    serial.sort_by_key(|mi| (mi.addr, mi.step));

    fs::create_dir_all(out)?;
    let mut rec = Vec::with_capacity(serial.len() * 32);
    for mi in &serial {
        rec.extend_from_slice(&(mi.addr as u64).to_le_bytes());
        rec.extend_from_slice(&mi.step.to_le_bytes());
        rec.extend_from_slice(&mi.value.to_le_bytes());
        rec.extend_from_slice(&(mi.is_write as u64).to_le_bytes());
    }
    fs::write(out.join("input_records.bin"), &rec)?;
    let meta = serde_json::json!({
        "chip": "mem",
        "case": "fullprogram",
        "source_elf": elf_path,
        "air": "Mem (air_id 14)",
        "segment_id": seg,
        "total_segments": total_segments,
        "input_count": serial.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "record_layout": "repr(C) { addr:u64, step:u64, value:u64, is_write:u64 } (native word addr; is_write 0/1)",
        "previous_segment": { "addr": prev.addr, "step": prev.step, "value": prev.value },
        "num_rows": num_rows,
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote mem seg {seg}/{total_segments}: {} ({rows} rows x {n_cols} cols, {} inputs)\n  golden_sha256: {golden_sha256}\n  prev_segment: addr={} step={} value={}",
        out.display(),
        serial.len(),
        prev.addr,
        prev.step,
        prev.value,
    );
    if debug_trace {
        write_trace_dump(&out.join("expected_mem_trace.npy.gz"), &air.trace, rows, n_cols)?;
        println!("  + expected_mem_trace.npy.gz (golden preimage, not committed)");
    }
    Ok(())
}

/// RomData (air_id 15) full-program oracle — the read-only ROM-data module.
/// Same count→plan→expand pipeline as `emit_mem`, filtered to the RomData plan.
/// RomData rides MEM_BUS_ID; for block 21740136 it is a single segment.
fn emit_rom_data(
    elf_path: &str,
    inputs_path: Option<&str>,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
    pctx: Arc<ProofCtx<F>>,
) -> Result<()> {
    let (rom, min_traces) = load_guest_min_traces(elf_path, inputs_path)?;

    // Count → plan (close() builds the per-module addr partitions; see --mem-spike).
    let mut metrics: Vec<(ChunkId, Box<dyn BusDeviceMetrics>)> = Vec::new();
    for (i, emu_trace) in min_traces.iter().enumerate() {
        let mut bus = MemCountBus { counters: MemCounters::new(), mem_writes_seen: 0 };
        // The executor seeds the FIRST chunk's counter with the memory-init
        // sections before busing it (execution/rust.rs, `is_first()`); the
        // planner's offsets under-allocate every initialized address without
        // this, and the trace rows for the init writes land as overwrites.
        if i == 0 {
            bus.counters.init_with_mem_sections(&rom as &dyn zisk_core::MemDataSection);
        }
        ZiskEmulator::process_emu_trace::<F, (), MemCountBus>(&rom, emu_trace, &mut bus, true);
        bus.counters.close();
        metrics.push((ChunkId(i), Box::new(bus.counters)));
    }
    let mut plans: Vec<_> = MemPlanner::new()
        .plan(metrics)
        .into_iter()
        .filter(|p| p.air_id == ROM_DATA_AIR_IDS[0])
        .collect();
    if plans.is_empty() {
        bail!("planner produced no RomData (air_id 15) plan");
    }
    plans.sort_by_key(|p| p.segment_id.map(|s| s.0).unwrap_or(0));
    let total_segments = plans.len();
    // RomData is single-segment on block 21740136; cap matches emit_mem so a
    // future multi-segment guest stays bounded.
    const MAX_ROM_DATA_SEGMENTS: usize = 2;

    let module = RomDataSM::new(std.clone());
    let (min_addr, _max_addr) = module.get_addr_range();
    for plan in plans.into_iter().take(MAX_ROM_DATA_SEGMENTS) {
        let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
        emit_one_rom_data_segment(
            &rom,
            &min_traces,
            plan,
            &out.join(format!("seg{seg:02}")),
            elf_path,
            debug_trace,
            min_addr,
            total_segments,
            std.clone(),
            &sctx,
            &pctx,
        )?;
    }
    Ok(())
}

/// Expand + emit one RomData segment — the RomData sibling of
/// `emit_one_mem_segment`. Differs only in the module (RomDataSM), the trace
/// shape (RomDataTrace, air_id 15, 2^21 rows), and the metadata labels; the
/// collector/sort/golden path is identical (RomData reuses MemInput).
#[allow(clippy::too_many_arguments)]
fn emit_one_rom_data_segment(
    rom: &ZiskRom,
    min_traces: &[zisk_common::EmuTrace],
    plan: zisk_common::Plan,
    out: &Path,
    elf_path: &str,
    debug_trace: bool,
    min_addr: u32,
    total_segments: usize,
    std: Arc<Std<F>>,
    sctx: &Arc<SetupCtx<F>>,
    pctx: &Arc<ProofCtx<F>>,
) -> Result<()> {
    let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
    let chunk_ids = plan_chunk_ids(&plan.check_point, "rom_data")?;
    let instance = MemModuleInstance::new(RomDataSM::new(std), InstanceCtx::new(0, plan));
    let mut collectors: Vec<(usize, Box<dyn BusDevice<u64>>)> = Vec::new();
    let mut serial: Vec<MemInput> = Vec::new();
    let mut prev = MemPreviousSegment { addr: min_addr, step: 0, value: 0 };
    for cid in &chunk_ids {
        let collector = instance.build_mem_collector(*cid, rom);
        let mut bus = MemFeedBus { collector };
        ZiskEmulator::process_emu_trace::<F, (), MemFeedBus>(rom, &min_traces[cid.0], &mut bus, true);
        let collector = bus.collector;
        for mi in &collector.inputs {
            serial.push(MemInput::new(mi.addr, mi.is_write, mi.step, mi.value));
        }
        if let Some(ps) = &collector.prev_segment {
            prev = MemPreviousSegment { addr: ps.addr, step: ps.step, value: ps.value };
        }
        collectors.push((cid.0, Box::new(collector) as Box<dyn BusDevice<u64>>));
    }

    let num_rows = RomDataTrace::<RomDataTraceRow<F>>::NUM_ROWS;
    let buffer = vec![F::default(); num_rows * RomDataTrace::<RomDataTraceRow<F>>::ROW_SIZE];
    let air = instance
        .compute_witness(pctx, sctx, collectors, buffer, false)?
        .ok_or_else(|| anyhow::anyhow!("RomData instance produced no AirInstance"))?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let golden_sha256 = trace_golden_sha256(&air.trace);

    // rw's BuildRomDataInputs consumes ops in (addr, step) order; mirror native
    // prepare_inputs so the records match the golden trace.
    serial.sort_by_key(|mi| (mi.addr, mi.step));

    fs::create_dir_all(out)?;
    let mut rec = Vec::with_capacity(serial.len() * 32);
    for mi in &serial {
        rec.extend_from_slice(&(mi.addr as u64).to_le_bytes());
        rec.extend_from_slice(&mi.step.to_le_bytes());
        rec.extend_from_slice(&mi.value.to_le_bytes());
        rec.extend_from_slice(&(mi.is_write as u64).to_le_bytes());
    }
    fs::write(out.join("input_records.bin"), &rec)?;
    let meta = serde_json::json!({
        "chip": "rom_data",
        "case": "fullprogram",
        "source_elf": elf_path,
        "air": "RomData (air_id 15)",
        "segment_id": seg,
        "total_segments": total_segments,
        "input_count": serial.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "record_layout": "repr(C) { addr:u64, step:u64, value:u64, is_write:u64 } (native word addr; is_write 0/1)",
        "previous_segment": { "addr": prev.addr, "step": prev.step, "value": prev.value },
        "num_rows": num_rows,
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote rom_data seg {seg}/{total_segments}: {} ({rows} rows x {n_cols} cols, {} inputs)\n  golden_sha256: {golden_sha256}\n  prev_segment: addr={} step={} value={}",
        out.display(),
        serial.len(),
        prev.addr,
        prev.step,
        prev.value,
    );
    if debug_trace {
        write_trace_dump(&out.join("expected_rom_data_trace.npy.gz"), &air.trace, rows, n_cols)?;
        println!("  + expected_rom_data_trace.npy.gz (golden preimage, not committed)");
    }
    Ok(())
}

/// InputData (air_id 16) full-program oracle — the read-only input-data module.
/// Same count→plan→expand pipeline as `emit_rom_data`, filtered to the InputData
/// plan. InputData rides MEM_BUS_ID; for block 21740136 it is a single segment.
fn emit_input_data(
    elf_path: &str,
    inputs_path: Option<&str>,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
    pctx: Arc<ProofCtx<F>>,
) -> Result<()> {
    let (rom, min_traces) = load_guest_min_traces(elf_path, inputs_path)?;

    // Count → plan (close() builds the per-module addr partitions; see --mem-spike).
    let mut metrics: Vec<(ChunkId, Box<dyn BusDeviceMetrics>)> = Vec::new();
    for (i, emu_trace) in min_traces.iter().enumerate() {
        let mut bus = MemCountBus { counters: MemCounters::new(), mem_writes_seen: 0 };
        // The executor seeds the FIRST chunk's counter with the memory-init
        // sections before busing it (execution/rust.rs, `is_first()`); the
        // planner's offsets under-allocate every initialized address without
        // this, and the trace rows for the init writes land as overwrites.
        if i == 0 {
            bus.counters.init_with_mem_sections(&rom as &dyn zisk_core::MemDataSection);
        }
        ZiskEmulator::process_emu_trace::<F, (), MemCountBus>(&rom, emu_trace, &mut bus, true);
        bus.counters.close();
        metrics.push((ChunkId(i), Box::new(bus.counters)));
    }
    let mut plans: Vec<_> = MemPlanner::new()
        .plan(metrics)
        .into_iter()
        .filter(|p| p.air_id == INPUT_DATA_AIR_IDS[0])
        .collect();
    if plans.is_empty() {
        bail!("planner produced no InputData (air_id 16) plan");
    }
    plans.sort_by_key(|p| p.segment_id.map(|s| s.0).unwrap_or(0));
    let total_segments = plans.len();
    // InputData is single-segment on block 21740136; cap matches emit_rom_data so
    // a future multi-segment guest stays bounded.
    const MAX_INPUT_DATA_SEGMENTS: usize = 2;

    let module = InputDataSM::new(std.clone());
    let (min_addr, _max_addr) = module.get_addr_range();
    for plan in plans.into_iter().take(MAX_INPUT_DATA_SEGMENTS) {
        let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
        emit_one_input_data_segment(
            &rom,
            &min_traces,
            plan,
            &out.join(format!("seg{seg:02}")),
            elf_path,
            debug_trace,
            min_addr,
            total_segments,
            std.clone(),
            &sctx,
            &pctx,
        )?;
    }
    Ok(())
}

/// Expand + emit one InputData segment — the InputData sibling of
/// `emit_one_rom_data_segment`. Differs only in the module (InputDataSM), the
/// trace shape (InputDataTrace, air_id 16, 2^21 rows), and the metadata labels;
/// the collector/sort/golden path is identical (InputData reuses MemInput).
#[allow(clippy::too_many_arguments)]
fn emit_one_input_data_segment(
    rom: &ZiskRom,
    min_traces: &[zisk_common::EmuTrace],
    plan: zisk_common::Plan,
    out: &Path,
    elf_path: &str,
    debug_trace: bool,
    min_addr: u32,
    total_segments: usize,
    std: Arc<Std<F>>,
    sctx: &Arc<SetupCtx<F>>,
    pctx: &Arc<ProofCtx<F>>,
) -> Result<()> {
    let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
    let chunk_ids = plan_chunk_ids(&plan.check_point, "input_data")?;
    let instance = MemModuleInstance::new(InputDataSM::new(std), InstanceCtx::new(0, plan));
    let mut collectors: Vec<(usize, Box<dyn BusDevice<u64>>)> = Vec::new();
    let mut serial: Vec<MemInput> = Vec::new();
    let mut prev = MemPreviousSegment { addr: min_addr, step: 0, value: 0 };
    for cid in &chunk_ids {
        let collector = instance.build_mem_collector(*cid, rom);
        let mut bus = MemFeedBus { collector };
        ZiskEmulator::process_emu_trace::<F, (), MemFeedBus>(rom, &min_traces[cid.0], &mut bus, true);
        let collector = bus.collector;
        for mi in &collector.inputs {
            serial.push(MemInput::new(mi.addr, mi.is_write, mi.step, mi.value));
        }
        if let Some(ps) = &collector.prev_segment {
            prev = MemPreviousSegment { addr: ps.addr, step: ps.step, value: ps.value };
        }
        collectors.push((cid.0, Box::new(collector) as Box<dyn BusDevice<u64>>));
    }

    let num_rows = InputDataTrace::<InputDataTraceRow<F>>::NUM_ROWS;
    let buffer = vec![F::default(); num_rows * InputDataTrace::<InputDataTraceRow<F>>::ROW_SIZE];
    let air = instance
        .compute_witness(pctx, sctx, collectors, buffer, false)?
        .ok_or_else(|| anyhow::anyhow!("InputData instance produced no AirInstance"))?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let golden_sha256 = trace_golden_sha256(&air.trace);

    // rw's BuildInputDataInputs consumes ops in (addr, step) order; mirror native
    // prepare_inputs so the records match the golden trace.
    serial.sort_by_key(|mi| (mi.addr, mi.step));

    fs::create_dir_all(out)?;
    let mut rec = Vec::with_capacity(serial.len() * 32);
    for mi in &serial {
        rec.extend_from_slice(&(mi.addr as u64).to_le_bytes());
        rec.extend_from_slice(&mi.step.to_le_bytes());
        rec.extend_from_slice(&mi.value.to_le_bytes());
        rec.extend_from_slice(&(mi.is_write as u64).to_le_bytes());
    }
    fs::write(out.join("input_records.bin"), &rec)?;
    let meta = serde_json::json!({
        "chip": "input_data",
        "case": "fullprogram",
        "source_elf": elf_path,
        "air": "InputData (air_id 16)",
        "segment_id": seg,
        "total_segments": total_segments,
        "input_count": serial.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "record_layout": "repr(C) { addr:u64, step:u64, value:u64, is_write:u64 } (native word addr; is_write 0/1)",
        "previous_segment": { "addr": prev.addr, "step": prev.step, "value": prev.value },
        "num_rows": num_rows,
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote input_data seg {seg}/{total_segments}: {} ({rows} rows x {n_cols} cols, {} inputs)\n  golden_sha256: {golden_sha256}\n  prev_segment: addr={} step={} value={}",
        out.display(),
        serial.len(),
        prev.addr,
        prev.step,
        prev.value,
    );
    if debug_trace {
        write_trace_dump(&out.join("expected_input_data_trace.npy.gz"), &air.trace, rows, n_cols)?;
        println!("  + expected_input_data_trace.npy.gz (golden preimage, not committed)");
    }
    Ok(())
}

/// Minimal `DataBusTrait` for the full-program oracle: forwards every
/// operation-bus write to the native per-chip collectors (each self-filters by
/// op_type and applies the native FROPS diversion), and tallies the raw op count
/// per family so the FROPS split (raw seen vs collected) is observable on a real
/// program. The `T` device type is unused — inputs are read back from the
/// collectors directly after the run.
struct BinaryOracleBus {
    basic: BinaryBasicCollector<F>,
    extension: BinaryExtensionCollector<F>,
    /// Arith (air_id 21) inputs. `ArithInstanceCollector.inputs` is private, so
    /// we replicate its tiny non-FROPS filter here and feed `ArithFullSM`
    /// `compute_witness` directly. Holds the raw bus `OperationData`.
    arith: Vec<OperationData<u64>>,
    /// FROPS table id for the Arith non-frequent-op test (mirrors the collector).
    arith_frops_table: usize,
    /// Sha256f (air_id 29) inputs. Mirrors the native `Sha256fCollector`: the bus
    /// data carries the state+input inline, so each op maps to one `Sha256fInput`.
    /// Precompiles have no FROPS diversion — every op becomes an input.
    sha256: Vec<Sha256fInput>,
    /// Keccakf (air_id 28) inputs. Same precompile shape as Sha256f: the bus data
    /// carries the 25-lane state inline, so each op maps to one `KeccakfInput`.
    keccak: Vec<KeccakfInput>,
    /// ArithEq (air_id 26) secp256k1_add (op 2) inputs. ArithEq multiplexes many
    /// curve ops into one air; the secp256k1-direct guest emits only
    /// secp256k1_add, so we collect just that variant. Like the other
    /// precompiles, no FROPS — every op is one input.
    secp256k1_add: Vec<Secp256k1AddInput>,
    /// ArithEq (air_id 26) secp256k1_dbl (op 3) inputs. The doubling sibling of
    /// `secp256k1_add`; the secp256k1-dbl-direct guest emits only this variant.
    secp256k1_dbl: Vec<Secp256k1DblInput>,
    /// ArithEq (air_id 26) secp256r1_add (op 9) inputs. The P-256 twin of
    /// `secp256k1_add`; the secp256r1-add-direct guest emits only this variant.
    secp256r1_add: Vec<Secp256r1AddInput>,
    /// ArithEq (air_id 26) secp256r1_dbl (op 10) inputs. The P-256 doubling twin;
    /// the secp256r1-dbl-direct guest emits only this variant.
    secp256r1_dbl: Vec<Secp256r1DblInput>,
    /// ArithEq (air_id 26) bn254_curve_add (op 4) inputs. The bn254-G1 twin of
    /// `secp256k1_add` (a = 0, distinct base field); the bn254-curve-add-direct
    /// guest emits only this variant.
    bn254_curve_add: Vec<Bn254CurveAddInput>,
    /// ArithEq (air_id 26) bn254_curve_dbl (op 5) inputs. The bn254-G1 doubling
    /// twin; the bn254-curve-dbl-direct guest emits only this variant.
    bn254_curve_dbl: Vec<Bn254CurveDblInput>,
    /// ArithEq (air_id 26) bn254_complex_add (op 6) inputs. Fp2 addition over the
    /// bn254 base field; the bn254-complex-add-direct guest emits only this
    /// variant. Same 152-byte two-operand record as the curve add, but the
    /// operands are field elements `f = (real ‖ imag)`, not curve points.
    bn254_complex_add: Vec<Bn254ComplexAddInput>,
    /// ArithEq (air_id 26) bn254_complex_sub (op 7) inputs. The Fp2 subtraction
    /// sibling; the bn254-complex-sub-direct guest emits only this variant.
    bn254_complex_sub: Vec<Bn254ComplexSubInput>,
    /// ArithEq (air_id 26) bn254_complex_mul (op 8) inputs. The Fp2 multiplication
    /// sibling; the bn254-complex-mul-direct guest emits only this variant.
    bn254_complex_mul: Vec<Bn254ComplexMulInput>,
    /// ArithEq (air_id 26) arith256 (op 0) inputs. The non-curve 256-bit
    /// multiply-accumulate `a*b + c = dh:dl`; the arith256-direct guest emits
    /// only this variant. Two 256-bit result halves, no modulus.
    arith256: Vec<Arith256Input>,
    /// ArithEq (air_id 26) arith256_mod (op 1) inputs. The modular sibling
    /// `d = (a*b + c) mod module`; the arith256-mod-direct guest emits only this
    /// variant. Carries the extra `module` operand and a single result `d`.
    arith256_mod: Vec<Arith256ModInput>,
    /// ArithEq384 (air_id 27, the *separate* 384-bit air) arith384_mod (op 0)
    /// inputs: `d = (a*b + c) mod module` over 6-limb operands. Rides a distinct
    /// op-type (ZiskOperationType::ArithEq384); the arith384-mod-direct guest
    /// emits only this variant.
    arith384_mod: Vec<Arith384ModInput>,
    /// ArithEq384 bls12_381_curve_add (op 1) inputs: BLS12-381 G1 point addition
    /// over 12-limb (two 6-limb coords) points. The bls12-381-curve-add-direct
    /// guest emits only this variant.
    bls12_381_curve_add: Vec<Bls12_381CurveAddInput>,
    /// ArithEq384 bls12_381_curve_dbl (op 2) inputs: the BLS12-381 G1 doubling
    /// twin; the bls12-381-curve-dbl-direct guest emits only this variant.
    bls12_381_curve_dbl: Vec<Bls12_381CurveDblInput>,
    /// ArithEq384 bls12_381_complex_add (op 3) inputs: Fp2 addition over the
    /// bls12-381 base field. Same 216-byte two-operand record as the curve add,
    /// but the operands are field elements f = (real ‖ imag), not curve points.
    bls12_381_complex_add: Vec<Bls12_381ComplexAddInput>,
    /// ArithEq384 bls12_381_complex_sub (op 4) inputs: the Fp2 subtraction sibling.
    bls12_381_complex_sub: Vec<Bls12_381ComplexSubInput>,
    /// ArithEq384 bls12_381_complex_mul (op 5) inputs: the Fp2 multiplication sibling.
    bls12_381_complex_mul: Vec<Bls12_381ComplexMulInput>,
    std: Arc<Std<F>>,
    /// Raw op counts per family, before FROPS diversion.
    binary_ops_seen: u64,
    binary_e_ops_seen: u64,
    arith_ops_seen: u64,
}

impl DataBusTrait<u64, ()> for BinaryOracleBus {
    fn write_to_bus(&mut self, bus_id: BusId, data: &[u64], _data_ext: &[u64]) -> bool {
        if bus_id == OPERATION_BUS_ID {
            // Each collector re-checks the op_type; we tally here to measure the
            // FROPS split. Forward to both binary collectors — the non-matching one
            // is a cheap early return inside its process_data.
            match data[1] {
                t if t == zisk_core::ZiskOperationType::Binary as u64 => self.binary_ops_seen += 1,
                t if t == zisk_core::ZiskOperationType::BinaryE as u64 => {
                    self.binary_e_ops_seen += 1
                }
                t if t == zisk_core::ZiskOperationType::Arith as u64 => {
                    self.arith_ops_seen += 1;
                    // Mirror ArithInstanceCollector: skip FROPS-diverted ops (they
                    // emit rows from a virtual table, not from collected inputs).
                    let frops_row = ArithFrops::get_row(data[OP] as u8, data[A], data[B]);
                    if frops_row != ArithFrops::NO_FROPS {
                        self.std.inc_virtual_row(self.arith_frops_table, frops_row as u64, 1);
                    } else if let Ok(ExtOperationData::OperationData(d)) = data.try_into() {
                        self.arith.push(d);
                    }
                }
                t if t == zisk_core::ZiskOperationType::Sha256 as u64 => {
                    // Precompile: no FROPS. The bus data carries state+input
                    // inline (values[7..19]), so mirror Sha256fCollector and map
                    // each op to one Sha256fInput. Fail loud on an unexpected
                    // variant — silently dropping an op yields a wrong golden
                    // that is hard to trace back (matches the native collector).
                    match data.try_into() {
                        Ok(ExtOperationData::OperationSha256Data(d)) => {
                            self.sha256.push(Sha256fInput::from(&d));
                        }
                        _ => panic!(
                            "Expected ExtOperationData::OperationSha256Data for Sha256 operation"
                        ),
                    }
                }
                t if t == zisk_core::ZiskOperationType::Keccak as u64 => {
                    // Precompile: no FROPS. The bus data carries the 25-lane state
                    // inline, so mirror KeccakfCollector and map each op to one
                    // KeccakfInput. Fail loud on an unexpected variant (see Sha256).
                    match data.try_into() {
                        Ok(ExtOperationData::OperationKeccakData(d)) => {
                            self.keccak.push(KeccakfInput::from(&d));
                        }
                        _ => panic!(
                            "Expected ExtOperationData::OperationKeccakData for Keccak operation"
                        ),
                    }
                }
                t if t == zisk_core::ZiskOperationType::ArithEq as u64 => {
                    // Precompile: no FROPS. ArithEq multiplexes several curve ops
                    // into one air; the supported full-program slices each run a
                    // single-op guest (secp256k1-direct emits only add,
                    // secp256k1-dbl-direct only dbl, the secp256r1 twins the P-256
                    // ops, the bn254 twins the bn254-G1 curve ops, and the bn254
                    // complex twins the Fp2 add/sub/mul ops), so we collect those
                    // variants and mirror the native ArithEqInstance dispatch
                    // (Secp256{k1,r1}{Add,Dbl}Input / Bn254Curve{Add,Dbl}Input /
                    // Bn254Complex{Add,Sub,Mul}Input ::from). A guest emitting any
                    // other ArithEq variant is unsupported here — fail loud rather
                    // than silently drop it into a wrong golden.
                    match data.try_into() {
                        Ok(ExtOperationData::OperationSecp256k1AddData(d)) => {
                            self.secp256k1_add.push(Secp256k1AddInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationSecp256k1DblData(d)) => {
                            self.secp256k1_dbl.push(Secp256k1DblInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationSecp256r1AddData(d)) => {
                            self.secp256r1_add.push(Secp256r1AddInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationSecp256r1DblData(d)) => {
                            self.secp256r1_dbl.push(Secp256r1DblInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBn254CurveAddData(d)) => {
                            self.bn254_curve_add.push(Bn254CurveAddInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBn254CurveDblData(d)) => {
                            self.bn254_curve_dbl.push(Bn254CurveDblInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBn254ComplexAddData(d)) => {
                            self.bn254_complex_add.push(Bn254ComplexAddInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBn254ComplexSubData(d)) => {
                            self.bn254_complex_sub.push(Bn254ComplexSubInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBn254ComplexMulData(d)) => {
                            self.bn254_complex_mul.push(Bn254ComplexMulInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationArith256Data(d)) => {
                            self.arith256.push(Arith256Input::from(&d));
                        }
                        Ok(ExtOperationData::OperationArith256ModData(d)) => {
                            self.arith256_mod.push(Arith256ModInput::from(&d));
                        }
                        _ => panic!(
                            "Expected ExtOperationData::OperationSecp256{{k1,r1}}{{Add,Dbl}}Data, \
                             OperationBn254Curve{{Add,Dbl}}Data, \
                             OperationBn254Complex{{Add,Sub,Mul}}Data, or \
                             OperationArith256{{,Mod}}Data; this oracle handles only the \
                             secp256k1 / secp256r1 / bn254_curve / bn254_complex / arith256 \
                             ArithEq ops"
                        ),
                    }
                }
                t if t == zisk_core::ZiskOperationType::ArithEq384 as u64 => {
                    // Precompile on the *separate* 384-bit air (air_id 27): no
                    // FROPS. arith384_mod is the only variant wired so far (the
                    // bls12-381 G1 curve / Fp2 ops share this air but ride their
                    // own op-data). Fail loud on any other ArithEq384 op — see
                    // the ArithEq arm.
                    match data.try_into() {
                        Ok(ExtOperationData::OperationArith384ModData(d)) => {
                            self.arith384_mod.push(Arith384ModInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBls12_381CurveAddData(d)) => {
                            self.bls12_381_curve_add.push(Bls12_381CurveAddInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBls12_381CurveDblData(d)) => {
                            self.bls12_381_curve_dbl.push(Bls12_381CurveDblInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBls12_381ComplexAddData(d)) => {
                            self.bls12_381_complex_add.push(Bls12_381ComplexAddInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBls12_381ComplexSubData(d)) => {
                            self.bls12_381_complex_sub.push(Bls12_381ComplexSubInput::from(&d));
                        }
                        Ok(ExtOperationData::OperationBls12_381ComplexMulData(d)) => {
                            self.bls12_381_complex_mul.push(Bls12_381ComplexMulInput::from(&d));
                        }
                        _ => panic!(
                            "Expected ExtOperationData::OperationArith384ModData, \
                             OperationBls12_381Curve{{Add,Dbl}}Data, or \
                             OperationBls12_381Complex{{Add,Sub,Mul}}Data; this oracle handles \
                             only the arith384_mod / bls12_381_curve_{{add,dbl}} / \
                             bls12_381_complex_{{add,sub,mul}} ArithEq384 ops"
                        ),
                    }
                }
                _ => {}
            }
            self.basic.process_data(&bus_id, data);
            self.extension.process_data(&bus_id, data);
        }
        true
    }

    fn on_close(&mut self) {}

    fn into_devices(self, _execute_on_close: bool) -> Vec<(usize, ())> {
        Vec::new()
    }
}

/// One binary operation in a fixture, mirrored byte-for-byte by the rw C++ record.
/// rw builds `ZiskStepData{opcode=op, rs1_val=a, rs2_val=b, rd_val=result}` from it.
#[repr(C)]
struct BinaryRecord {
    op: u8,
    _pad: [u8; 7],
    a: u64,
    b: u64,
    result: u64,
}

/// Stands up the proofman setup context from the installed ZisK proving key.
/// CPU-only (`gpu = false`). Returns the `Std` the state machines need plus the
/// `SetupCtx` it was built from — most SMs use `Std` alone, but
/// `ArithEqSM::compute_witness` also takes a `&SetupCtx`, so the caller keeps
/// the Arc alive and threads it in.
fn build_std(hash_family: Option<&str>) -> Result<(Arc<Std<F>>, Arc<SetupCtx<F>>, Arc<ProofCtx<F>>)> {
    let gpu = false;
    let proving_key = ZiskPaths::get_proving_key(None);
    if !proving_key.exists() {
        bail!(
            "proving key not found at {} — run `~/.zisk/bin/ziskup --provingkey`",
            proving_key.display()
        );
    }
    let proving_key = match hash_family {
        Some(family) => shadow_proving_key(&proving_key, family)?,
        None => proving_key,
    };
    let mpi_ctx = Arc::new(MpiCtx::new());
    let mut pctx = ProofCtx::<F>::create_ctx(proving_key, false, VerboseMode::Info, mpi_ctx, gpu)?;
    let sctx = Arc::new(SetupCtx::<F>::new(&pctx.global_info, &ProofType::Basic, false, &[], gpu)?);
    let setups_vadcop = Arc::new(SetupsVadcop::new(&pctx.global_info, false, false, &[], gpu)?);
    init_gpu_setup(sctx.max_n_bits_ext as u64, gpu)?;
    pctx.set_device_buffers(&sctx, &setups_vadcop, false, gpu, 1)?;
    // Instance registration (--stage1-root) weighs instances via dctx; the
    // weights map is only filled on demand, mirroring ProofMan::new.
    pctx.set_weights(&sctx)?;
    let pctx = Arc::new(pctx);
    let std = Std::new(pctx.clone(), sctx.clone(), false)?;
    Ok((std, sctx, pctx))
}

/// The tree hash family is set ONCE per process, inside `GlobalInfo::load`
/// (`define_hash_family` throws on any later change) — so `--hash-family`
/// cannot be applied after the setup context exists. Instead, mirror the
/// proving key into a temp dir of symlinks with only `pilout.globalInfo.json`
/// rewritten to carry the requested family, and load that. The installed key
/// is never touched.
fn shadow_proving_key(real: &Path, family: &str) -> Result<PathBuf> {
    if !proofman_common::is_known_family(family) {
        bail!("--hash-family {family:?} unknown; known families: {:?}", proofman_common::FAMILIES);
    }
    let shadow = std::env::temp_dir().join(format!("rw-fixture-gen-key-{family}"));
    if shadow.exists() {
        fs::remove_dir_all(&shadow)?;
    }
    fs::create_dir_all(&shadow)?;
    for entry in fs::read_dir(real)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "pilout.globalInfo.json" {
            let mut gi: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(entry.path())?)?;
            gi["hash"] = serde_json::Value::String(family.to_string());
            fs::write(shadow.join(&name), serde_json::to_string_pretty(&gi)?)?;
        } else {
            std::os::unix::fs::symlink(entry.path(), shadow.join(&name))?;
        }
    }
    Ok(shadow)
}

/// Decompress a `--debug-trace` dump and return its payload as canonical u64s.
/// Only the v1 npy header this tool's `write_npy_gz` emits is parsed; the
/// payload bytes after the header are exactly the `golden_sha256` preimage.
fn read_npy_gz(path: &Path, rows: usize, cols: usize) -> Result<Vec<u64>> {
    use std::io::Read;
    let mut raw = Vec::new();
    GzDecoder::new(fs::File::open(path).with_context(|| format!("opening {}", path.display()))?)
        .read_to_end(&mut raw)?;
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" || raw[6] != 1 {
        bail!("{} is not a v1 .npy this tool wrote", path.display());
    }
    let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let payload = &raw[10 + header_len..];
    if payload.len() != rows * cols * 8 {
        bail!(
            "{}: payload is {} bytes, metadata says {rows} rows x {cols} cols ({} bytes)",
            path.display(),
            payload.len(),
            rows * cols * 8
        );
    }
    Ok(payload.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Stage-1 commit oracle (zisk-zorch #83): reload a dumped trace into an
/// `AirInstance` registered on the proof context and run the native stage-1
/// commit on it via proofman's contribution path — `commit_witness` =
/// witness-expr hints + coset LDE + `MerkleTreeGL` — the same path a real
/// prove takes. Emits `stage1_commit.json` next to the fixture with the root
/// and the starkStruct params. `committed_trace_sha256` is rehashed AFTER the
/// commit: if it differs from the fixture's `golden_sha256`, the AIR has
/// `witness_calc` hints that rewrite trace columns before the LDE, and a
/// downstream byte-match must reproduce that step too.
fn emit_stage1_root(fixture_dir: &Path, sctx: &Arc<SetupCtx<F>>, pctx: &Arc<ProofCtx<F>>) -> Result<()> {
    // The family is fixed at GlobalInfo::load (set-once per process); a
    // --hash-family override reached it via build_std's shadow proving key.
    let hash_family = pctx.global_info.hash.clone();
    let meta_path = fixture_dir.join("fixture_metadata.json");
    let meta: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&meta_path).with_context(|| format!("reading {}", meta_path.display()))?,
    )?;
    let air_label = meta["air"].as_str().context("metadata has no \"air\" field")?;
    let rows = meta["trace_rows"].as_u64().context("metadata has no trace_rows")? as usize;
    let cols = meta["trace_cols"].as_u64().context("metadata has no trace_cols")? as usize;
    let golden_sha256 = meta["golden_sha256"].as_str().context("metadata has no golden_sha256")?;
    let air_id: usize = air_label
        .split("air_id ")
        .nth(1)
        .map(|s| s.chars().take_while(char::is_ascii_digit).collect::<String>())
        .and_then(|d| d.parse().ok())
        .with_context(|| format!("cannot parse air_id from {air_label:?}"))?;
    // All ZisK basic AIRs live in the single "Zisk" airgroup.
    let airgroup_id = 0usize;

    let dump = fs::read_dir(fixture_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                n.starts_with("expected_") && n.ends_with("_trace.npy.gz")
            })
        })
        .collect::<Vec<_>>();
    let [dump] = dump.as_slice() else {
        bail!("expected exactly one expected_*_trace.npy.gz in {} (run with --debug-trace first)", fixture_dir.display());
    };

    let words = read_npy_gz(dump, rows, cols)?;
    let mut hasher = Sha256::new();
    for w in &words {
        hasher.update(w.to_le_bytes());
    }
    let loaded_sha = hex(&hasher.finalize());
    if loaded_sha != golden_sha256 {
        bail!("trace dump hash {loaded_sha} != metadata golden_sha256 {golden_sha256} — stale dump?");
    }
    let trace: Vec<F> = words.into_iter().map(F::from_u64).collect();

    // commit_witness reads the trace at starkinfo's cm1 width; a dump narrower
    // than cm1 means this fork's SM/pil helpers have drifted from the installed
    // proving key for this AIR (e.g. RomData: SM writes 5 cols, alpha key's cm1
    // is 6 incl `sel`) — committing would misread every row, so refuse.
    let setup = sctx
        .get_setup(airgroup_id, air_id)
        .map_err(|e| anyhow::anyhow!("get_setup: {e:?}"))?;
    let cm1_cols = *setup
        .stark_info
        .map_sections_n
        .get("cm1")
        .context("starkinfo has no cm1 section")? as usize;
    if cm1_cols != cols {
        bail!(
            "{air_label}: dumped trace has {cols} cols but the proving key's cm1 is {cm1_cols} — \
             SM/proving-key drift; this AIR cannot be committed against the installed key"
        );
    }

    // Register the instance the way the witness executor does, then commit it
    // through the same path a real prove uses for its stage-1 contributions.
    // Single-process partition layout, as ProofMan sets before a local prove.
    pctx.dctx_setup(1, vec![0], 0).map_err(|e| anyhow::anyhow!("dctx_setup: {e:?}"))?;
    let global_idx = pctx
        .add_instance_assign(airgroup_id, air_id)
        .map_err(|e| anyhow::anyhow!("add_instance_assign: {e:?}"))?;
    // Airvalues (e.g. a mem-family segment carry) ride the contribution hash,
    // not the stage-1 root, so zeros keep the commit well-formed without
    // affecting the golden. Sized like proofman's initialize_air_instance.
    let n_airvalues = setup
        .stark_info
        .airvalues_map
        .as_ref()
        .map(|m| m.iter().map(|e| if e.stage == 1 { 1 } else { 3 }).sum::<usize>())
        .unwrap_or(0);
    pctx.add_air_instance(
        AirInstance::new(
            TraceInfo::new(airgroup_id, air_id, cols, rows, trace, false, false)
                .with_air_values(vec![F::default(); n_airvalues]),
        ),
        global_idx,
    );

    let mut aux_trace = vec![F::default(); sctx.max_prover_buffer_size];
    let mut const_pols = vec![F::default(); sctx.max_const_size];
    let roots: Vec<[F; 4]> = vec![[F::default(); 4]; global_idx + 1];
    let values: Vec<std::sync::Mutex<Vec<F>>> =
        (0..=global_idx).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    ProofMan::<F>::get_contribution_air(
        pctx,
        sctx,
        &roots,
        &values,
        global_idx,
        &mut aux_trace,
        &mut const_pols,
    )
    .map_err(|e| anyhow::anyhow!("get_contribution_air: {e:?}"))?;
    let root: Vec<String> = roots[global_idx].iter().map(|f| f.as_canonical_u64().to_string()).collect();

    // Rehash the (possibly witness-expr-rewritten) committed trace.
    let steps = pctx.get_air_instance_params(global_idx, false);
    let committed = unsafe { std::slice::from_raw_parts(steps.trace as *const F, rows * cols) };
    let mut hasher = Sha256::new();
    for f in committed {
        hasher.update(f.as_canonical_u64().to_le_bytes());
    }
    let committed_sha = hex(&hasher.finalize());

    let ss = &setup.stark_info.stark_struct;
    let doc = serde_json::json!({
        "air": air_label,
        "airgroup_id": airgroup_id,
        "air_id": air_id,
        "n_bits": ss.n_bits,
        "n_bits_ext": ss.n_bits_ext,
        "blowup_bits": ss.n_bits_ext - ss.n_bits,
        "merkle_tree_arity": ss.merkle_tree_arity,
        // The tree/leaf hash family the root was computed with — the proving
        // key's globalInfo (absent field => proofman's default, currently
        // Poseidon1) unless --hash-family overrode it. A downstream
        // byte-match MUST use the same family or the root is unreproducible.
        "hash_family": hash_family,
        "trace_rows": rows,
        "trace_cols": cols,
        "field": "goldilocks_canonical_u64",
        "trace_sha256": golden_sha256,
        "committed_trace_sha256": committed_sha,
        "root": root,
        "root_semantics": "pil2-stark stage-1 commit (commit_witness): witness_calc hints, coset-SHIFT LDE to 2^n_bits_ext, linear-hash leaves, arity-k MerkleTreeGL in hash_family",
    });
    let out = fixture_dir.join("stage1_commit.json");
    fs::write(&out, serde_json::to_string_pretty(&doc)?)?;
    println!(
        "wrote {}: root = [{}]\n  {air_label}: n_bits={} n_bits_ext={} arity={} cols={cols}\n  committed trace {} golden preimage",
        out.display(),
        root.join(", "),
        ss.n_bits,
        ss.n_bits_ext,
        ss.merkle_tree_arity,
        if committed_sha == golden_sha256 { "==" } else { "DIFFERS FROM" },
    );
    Ok(())
}

/// Native 64-bit result for a binary op. Mirrors `BinaryBasicSM::execute`:
/// ADD/SUB wrap modulo 2^64.
fn binary_result(op: u8, a: u64, b: u64) -> Result<u64> {
    match op {
        x if x == ADD_OP => Ok(a.wrapping_add(b)),
        x if x == SUB_OP => Ok(a.wrapping_sub(b)),
        // Comparators (results mirror op_* in zisk_ops.rs / gt_execute in
        // binary_basic.rs): 1 when the predicate holds, else 0. LTU/LEU compare
        // as unsigned, LT/GT/LE as signed (i64).
        x if x == EQ_OP => Ok(u64::from(a == b)),
        x if x == LTU_OP => Ok(u64::from(a < b)),
        x if x == LT_OP => Ok(u64::from((a as i64) < (b as i64))),
        x if x == GT_OP => Ok(u64::from((a as i64) > (b as i64))),
        x if x == LEU_OP => Ok(u64::from(a <= b)),
        x if x == LE_OP => Ok(u64::from((a as i64) <= (b as i64))),
        // MIN/MAX select a or b (not a 0/1 flag); MINU/MAXU unsigned, MIN/MAX
        // signed (op_min* in zisk_ops.rs).
        x if x == MINU_OP => Ok(if a < b { a } else { b }),
        x if x == MIN_OP => Ok(if (a as i64) < (b as i64) { a } else { b }),
        x if x == MAXU_OP => Ok(if a > b { a } else { b }),
        x if x == MAX_OP => Ok(if (a as i64) > (b as i64) { a } else { b }),
        // LtAbs (lt_abs_np/pn_execute in binary_basic.rs): NP negates a (-a via
        // two's complement) then compares -a < b unsigned; PN negates b and
        // compares a < -b. Used for absolute-value comparisons.
        x if x == LT_ABS_NP_OP => Ok(u64::from(a.wrapping_neg() < b)),
        x if x == LT_ABS_PN_OP => Ok(u64::from(a < b.wrapping_neg())),
        // Logic ops keep the bitwise result in free_in_c.
        x if x == AND_OP => Ok(a & b),
        x if x == OR_OP => Ok(a | b),
        x if x == XOR_OP => Ok(a ^ b),
        // W-variants (32-bit ops on the lower halves). Arithmetic results are
        // computed in i32 then SIGN-EXTENDED to i64 (Rust `i32 as i64 as u64`);
        // comparator results are 0/1; MIN/MAX W returns the selected operand's
        // lower 32 bits sign-extended (even MINUW/MAXUW sign-extend the result,
        // matching native op_*_w in zisk_ops.rs L579/603/714/765/782/833/850 and
        // L1150/1166/1214/1230).
        x if x == ADDW_OP => Ok((a as i32).wrapping_add(b as i32) as i64 as u64),
        x if x == SUBW_OP => Ok((a as i32).wrapping_sub(b as i32) as i64 as u64),
        x if x == EQW_OP => Ok(u64::from((a as i32) == (b as i32))),
        x if x == LTUW_OP => Ok(u64::from((a as u32) < (b as u32))),
        x if x == LTW_OP => Ok(u64::from((a as i32) < (b as i32))),
        x if x == LEUW_OP => Ok(u64::from((a as u32) <= (b as u32))),
        x if x == LEW_OP => Ok(u64::from((a as i32) <= (b as i32))),
        x if x == MINUW_OP => Ok(if (a as u32) < (b as u32) {
            a as i32 as i64 as u64
        } else {
            b as i32 as i64 as u64
        }),
        x if x == MINW_OP => Ok(if (a as i32) < (b as i32) {
            a as i32 as i64 as u64
        } else {
            b as i32 as i64 as u64
        }),
        x if x == MAXUW_OP => Ok(if (a as u32) > (b as u32) {
            a as i32 as i64 as u64
        } else {
            b as i32 as i64 as u64
        }),
        x if x == MAXW_OP => Ok(if (a as i32) > (b as i32) {
            a as i32 as i64 as u64
        } else {
            b as i32 as i64 as u64
        }),
        _ => bail!("unsupported op 0x{op:02x} (ADD/SUB/EQ/LT/MIN/MAX/LtAbs/logic + W wired)"),
    }
}

/// Native 64-bit result for a BinaryExtension op. Mirrors `op_*` in
/// zisk_ops.rs — the binary-extension SM itself does NOT carry a result
/// column, so this is purely for fixture record parity / debuggability (the
/// rw filler reads opcode/a/b and recomputes the per-byte trace cells).
fn binary_extension_result(op: u8, a: u64, b: u64) -> Result<u64> {
    // 64-bit shifts mask the shift amount to 6 bits (LS_6_BITS), W shifts
    // mask to 5 bits and sign-extend the 32-bit result back to i64 — see
    // `op_*` in zisk_ops.rs. SE ops sign-extend `b` (= input.b post the SM's
    // a/b swap) at byte / halfword / word width.
    let s6 = (b & 0x3F) as u32;
    let s5 = (b & 0x1F) as u32;
    let a32 = a as u32;
    match op {
        x if x == SLL_OP => Ok(a.wrapping_shl(s6)),
        x if x == SRL_OP => Ok(a.wrapping_shr(s6)),
        x if x == SRA_OP => Ok((a as i64).wrapping_shr(s6) as u64),
        x if x == SLLW_OP => Ok(a32.wrapping_shl(s5) as i32 as i64 as u64),
        x if x == SRLW_OP => Ok(a32.wrapping_shr(s5) as i32 as i64 as u64),
        x if x == SRAW_OP => Ok((a as i32).wrapping_shr(s5) as i64 as u64),
        x if x == SEXT_B_OP => Ok((b as i8) as i64 as u64),
        x if x == SEXT_H_OP => Ok((b as i16) as i64 as u64),
        x if x == SEXT_W_OP => Ok((b as i32) as i64 as u64),
        _ => bail!(
            "unsupported binary_extension op 0x{op:02x} \
             (BinaryExtension SM has 9 opcodes wired in this lineage)"
        ),
    }
}

/// Builds a BinaryBasic record, computing the native result for `op(a, b)`.
fn record(op: u8, a: u64, b: u64) -> Result<BinaryRecord> {
    Ok(BinaryRecord { op, _pad: [0; 7], a, b, result: binary_result(op, a, b)? })
}

/// Builds a BinaryExtension record.
fn record_extension(op: u8, a: u64, b: u64) -> Result<BinaryRecord> {
    Ok(BinaryRecord {
        op,
        _pad: [0; 7],
        a,
        b,
        result: binary_extension_result(op, a, b)?,
    })
}

/// Returns the records for a named case.
fn case_inputs(case: &str) -> Result<Vec<BinaryRecord>> {
    match case {
        "add_single" => Ok(vec![record(ADD_OP, 10, 20)?]),
        // SUB exercises the borrow chain and the carry_byte=7 forcing:
        //   30 - 20 = 10        : no borrow (carry all 0)
        //   0x100 - 1 = 0xFF    : single byte-0 borrow that clears at byte 1
        //   20 - 30 = wrap      : full borrow chain; cout at byte 7 forced to 0
        "sub_single" => {
            Ok(vec![record(SUB_OP, 30, 20)?, record(SUB_OP, 0x100, 1)?, record(SUB_OP, 20, 30)?])
        }
        // EQ exercises the comparator compare-chain and the byte-7 inversion, and
        // checks free_in_c is zeroed (native zeroes it for every comparator):
        //   5 == 5                    : equal  (result 1, carry [0..6]=0, byte-7 inverted to 1)
        //   5 != 7 (byte 0 differs)   : differ (result 0, carry propagates 1, byte-7 inverted to 0)
        //   0x100 != 0x200 (byte 1)   : differ in a higher byte (cin propagation)
        //   u64::MAX == u64::MAX      : equal across all bytes
        "eq_single" => Ok(vec![
            record(EQ_OP, 5, 5)?,
            record(EQ_OP, 5, 7)?,
            record(EQ_OP, 0x100, 0x200)?,
            record(EQ_OP, u64::MAX, u64::MAX)?,
        ]),
        // LT family exercises the per-opcode compare chains, the zeroed
        // free_in_c, the last-byte sign-disagreement override (LT/GT/LE), and
        // the LEU b_op remap (Leu == LEUW_OP == 0x1c):
        //   LTU: a<b, a>b, and a higher-byte differ (byte-0 equal → cin carries)
        //   LT : -1<1 (sign disagree, cout=sign(a)=1), 1<-1 (cout=sign(a)=0),
        //        -5<-2 (same sign, normal chain)
        //   GT : 1>-1 (sign disagree, cout=sign(b)=1), -1>1 (cout=sign(b)=0),
        //        10>5 (positive)
        //   LEU: 5<=5 (equal → carry all 1), 10<=5 (per-byte, byte-0 cout=0)
        //   LE : 5<=5 (equal), -1<=1 (sign disagree, cout=c=1), 1<=-1 (cout=c=0)
        "lt_single" => Ok(vec![
            record(LTU_OP, 5, 10)?,
            record(LTU_OP, 10, 5)?,
            record(LTU_OP, 0x100, 0x200)?,
            record(LT_OP, (-1i64) as u64, 1)?,
            record(LT_OP, 1, (-1i64) as u64)?,
            record(LT_OP, (-5i64) as u64, (-2i64) as u64)?,
            record(GT_OP, 1, (-1i64) as u64)?,
            record(GT_OP, (-1i64) as u64, 1)?,
            record(GT_OP, 10, 5)?,
            record(LEU_OP, 5, 5)?,
            record(LEU_OP, 10, 5)?,
            record(LE_OP, 5, 5)?,
            record(LE_OP, (-1i64) as u64, 1)?,
            record(LE_OP, 1, (-1i64) as u64)?,
        ]),
        // MIN/MAX keep the real result in free_in_c and set result_is_a
        // (=1 iff the result is a, i.e. a!=b && b!=c) and c_is_signed (real
        // result sign bit). Cases exercise result_is_a ∈ {0 via a==b, 0 via
        // b==c, 1}, c_is_signed ∈ {0, 1}, and signed vs unsigned selection:
        //   MINU 5,10→5 (ria=1); 10,5→5 (ria=0, b==c); 7,7→7 (ria=0, a==b)
        //   MINU MAX,0x80..00 → 0x80..00 (result sign bit set → c_is_signed=1)
        //   MIN  -1,1→-1 (signed, result=a, c_is_signed=1); 1,-1→-1 (b==c)
        //   MAXU 5,10→10 (ria=0, b==c); 10,5→10 (ria=1)
        //   MAX  -1,1→1 (result=b, c_is_signed=0); 1,-1→1 (ria=1)
        "minmax_single" => Ok(vec![
            record(MINU_OP, 5, 10)?,
            record(MINU_OP, 10, 5)?,
            record(MINU_OP, 7, 7)?,
            record(MINU_OP, u64::MAX, 0x8000_0000_0000_0000)?,
            record(MIN_OP, (-1i64) as u64, 1)?,
            record(MIN_OP, 1, (-1i64) as u64)?,
            record(MAXU_OP, 5, 10)?,
            record(MAXU_OP, 10, 5)?,
            record(MAX_OP, (-1i64) as u64, 1)?,
            record(MAX_OP, 1, (-1i64) as u64)?,
        ]),
        // LtAbs zeroes free_in_c and sets use_first_byte; the carry is a signed
        // per-byte subtraction sign chain over the complemented operand (byte-0
        // gets the two's-complement +1). NP compares -a < b, PN compares a < -b.
        // Cases cover less/greater/equal (sub==0 → cin) and multi-byte cin:
        //   NP -3,5→1 (3<5); -10,5→0 (10<5 no); -5,5→0 (equal); -0x100,0x200→1;
        //      -0x300,0x100→0
        //   PN 3,-5→1 (3<5); 10,-5→0; 5,-5→0 (equal); 0x100,-0x200→1;
        //      0x300,-0x100→0
        "ltabs_single" => Ok(vec![
            record(LT_ABS_NP_OP, (-3i64) as u64, 5)?,
            record(LT_ABS_NP_OP, (-10i64) as u64, 5)?,
            record(LT_ABS_NP_OP, (-5i64) as u64, 5)?,
            record(LT_ABS_NP_OP, (-0x100i64) as u64, 0x200)?,
            record(LT_ABS_NP_OP, (-0x300i64) as u64, 0x100)?,
            record(LT_ABS_PN_OP, 3, (-5i64) as u64)?,
            record(LT_ABS_PN_OP, 10, (-5i64) as u64)?,
            record(LT_ABS_PN_OP, 5, (-5i64) as u64)?,
            record(LT_ABS_PN_OP, 0x100, (-0x200i64) as u64)?,
            record(LT_ABS_PN_OP, 0x300, (-0x100i64) as u64)?,
        ]),
        // Logic (AND/OR/XOR) keeps the bitwise result in free_in_c but native
        // zeroes the carry on every byte and forces c_is_signed=false. Cases
        // include byte sums that overflow (0xFF+0xFF) — which the old add-carry
        // fall-through would have emitted as a spurious carry — and high-bit-set
        // results (byte 7 >= 0x80) to lock the c_is_signed=0 override:
        //   AND 0xFF,0x0F→0x0F; AND MAX,MAX→MAX (byte overflow + high bit)
        //   OR 0xFF00,0x00FF→0xFFFF; OR 0xFF,0xFF→0xFF (byte-0 overflow)
        //   XOR 0xAA,0x55→0xFF; XOR MAX,0x7FFF..→0x8000.. (high-bit result)
        "logic_single" => Ok(vec![
            record(AND_OP, 0xFF, 0x0F)?,
            record(AND_OP, u64::MAX, u64::MAX)?,
            record(OR_OP, 0xFF00, 0x00FF)?,
            record(OR_OP, 0xFF, 0xFF)?,
            record(XOR_OP, 0xAA, 0x55)?,
            record(XOR_OP, u64::MAX, 0x7FFF_FFFF_FFFF_FFFF)?,
        ]),
        // W-variants exercise the mode32 carry_byte=3 path AND the b_op remap
        // (each W op echoes its 64-bit sibling's table-op discriminant —
        // AddW→Add=0x0a, EqW→Eq=0x09, LtuW→Ltu=0x06, etc.; both LEU and LeuW
        // emit 0x1c because Leu's discriminant == LEUW_OP). Cases:
        //   arith W: ADDW with positive + overflow-to-negative (i32 sign bit
        //     set → result sign-extended to upper bytes 0xFF); SUBW likewise.
        //   comparator W: EQW where low 32 match but full 64 differ (1);
        //     LTUW/LTW (signed vs unsigned divergence); LEUW.
        //   min/max W: MINUW where unsigned-min has high bit set (result is
        //     sign-extended → upper bytes 0xFF, exercising the i32→i64 cast in
        //     the rd_val supplied by op_*_w); MINW signed; MAXUW; MAXW.
        "w_single" => Ok(vec![
            record(ADDW_OP, 5, 3)?,
            record(ADDW_OP, 0x7FFF_FFFF, 1)?,
            record(SUBW_OP, 10, 5)?,
            record(SUBW_OP, 5, 10)?,
            record(EQW_OP, 0xDEAD_BEEF_1234_5678, 0xCAFE_BABE_1234_5678)?,
            record(EQW_OP, 5, 7)?,
            record(LTUW_OP, 5, 10)?,
            record(LTW_OP, (-1i64) as u64, 1)?,
            record(LEUW_OP, 5, 5)?,
            record(LEW_OP, (-5i64) as u64, (-2i64) as u64)?,
            record(MINUW_OP, 0xFFFF_FFFF, 0x8000_0000)?,
            record(MINW_OP, (-5i64) as u64, 5)?,
            record(MAXUW_OP, 10, 5)?,
            record(MAXW_OP, (-1i64) as u64, 1)?,
        ]),
        other => bail!("unknown case {other:?}"),
    }
}

/// Returns the records for a BinaryExtension named case.
fn case_inputs_extension(case: &str) -> Result<Vec<BinaryRecord>> {
    match case {
        // SLL_OP=0x21. The single record exercises a non-trivial cross-byte
        // shift (b=4 carries data from byte j into the high nibble of byte
        // j+1 AND straddles the lo32/hi32 split at j=3→4). The rw filler's
        // SllByteShiftsKnownAnswer smoke test uses the SAME (a, b) so the
        // golden SHA pinned here cross-locks against the hand-derived smoke
        // expectations.
        "sll_single" => Ok(vec![record_extension(SLL_OP, 0xCAFEBABE_DEADBEEF, 4)?]),
        // Covers the remaining five shifts in a single fixture. Each record
        // matches its rw smoke test:
        //   SRL  : the SRL formula on the same a as SLL's known-answer test
        //   SRA  : a_bytes[7]=0xFF + b≠0 → the j==7 sign-fill path
        //   SLLW : a_bytes[3]=0xC0 + b=1 → bit 31 set after shift → SE
        //   SRLW : all 1s + b=0 → bit-31 SE at j==3, j>=4 gated to 0
        //   SRAW : a_bytes[3]=0x80 + b=4 → j==3 sign-fill via (32-b_low)
        "shifts_single" => Ok(vec![
            record_extension(SRL_OP, 0xCAFEBABE_DEADBEEF, 4)?,
            record_extension(SRA_OP, 0xFF00_0000_0000_0000, 4)?,
            record_extension(SLLW_OP, 0xC000_0000, 1)?,
            record_extension(SRLW_OP, 0xFFFF_FFFF_FFFF_FFFF, 0)?,
            record_extension(SRAW_OP, 0x8000_0000, 4)?,
        ]),
        // BinaryExtension SE ops swap inputs (a_val=input.b, b_val=input.a),
        // so the SE source operand lives in the record's `b` field. Records
        // mirror the rw filler's SE smoke tests:
        //   SE_B  b=0x80          — byte with sign bit → 0xFFFFFFFFFFFFFF80
        //   SE_H  b=0x8042        — halfword with sign bit → 0xFFFFFFFFFFFF8042
        //   SE_W  b=0x80123456    — word with sign bit → 0xFFFFFFFF80123456
        // a is set to a recognizable pattern so the b column (which holds
        // post-swap b_val = input.a for SE) is visually distinct in dumps.
        "sign_extend_single" => Ok(vec![
            record_extension(SEXT_B_OP, 0xDEADBEEFCAFE0000, 0x80)?,
            record_extension(SEXT_H_OP, 0xDEADBEEFCAFE0000, 0x8042)?,
            record_extension(SEXT_W_OP, 0xDEADBEEFCAFE0000, 0x80123456)?,
        ]),
        other => bail!("unknown binary_extension case {other:?}"),
    }
}

/// Writes a numpy `.npy` (v1.0, little-endian u64) gzipped to `path`.
fn write_npy_gz(path: &Path, data: &[u64], rows: usize, cols: usize) -> Result<()> {
    let mut header =
        format!("{{'descr': '<u8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // pad so (10 + header.len()+1) is a multiple of 64, then newline.
    let total = 10 + header.len() + 1;
    let pad = (64 - (total % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');

    let enc = GzEncoder::new(fs::File::create(path)?, Compression::default());
    let mut w = std::io::BufWriter::new(enc);
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn emit_binary(case: &str, out: &Path, debug_trace: bool, std: Arc<Std<F>>) -> Result<()> {
    let records = case_inputs(case)?;
    let inputs: Vec<Vec<BinaryInput>> =
        vec![records.iter().map(|r| BinaryInput::new(r.op, r.a, r.b)).collect()];

    let sm = BinaryBasicSM::<F>::new(std);
    let num_rows = BinaryTrace::<BinaryTraceRow<F>>::NUM_ROWS;
    let row_size = BinaryTrace::<BinaryTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<BinaryTraceRow<F>>(&inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    // Golden hash over the canonical trace serialization (row-major u64, little-endian).
    // This is the committed reference — NOT the trace blob. rw's zisk_trace_dump hashes
    // the same canonical form and compares.
    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;

    // Committed: tiny input records (#[repr(C)] BinaryRecord array, little-endian).
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.push(r.op);
        rec_bytes.extend_from_slice(&r._pad);
        rec_bytes.extend_from_slice(&r.a.to_le_bytes());
        rec_bytes.extend_from_slice(&r.b.to_le_bytes());
        rec_bytes.extend_from_slice(&r.result.to_le_bytes());
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    // Committed: metadata with the golden hash (no trace blob).
    let meta = serde_json::json!({
        "zisk_commit": "790f9e28a (fractalyze/zisk ref)",
        "chip": "binary",
        "case": case,
        "air": "BinaryBasic (air_id 22)",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { op:u8, pad[7], a:u64, b:u64, result:u64 }",
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;

    // Local debug only (NOT committed): full trace for first-diff localization.
    if debug_trace {
        write_npy_gz(&out.join("expected_binary_trace.npy.gz"), &data, rows, n_cols)?;
    }

    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_binary_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

/// Serialize one chip's collected full-program inputs into the standard fixture
/// contract — `input_records.bin` (`repr(C) { op:u8, pad[7], a:u64, b:u64,
/// result:u64 }`, the same layout the per-op `emit_*` paths write, so rw reads
/// full-program records through the identical reader) plus `fixture_metadata.json`
/// carrying the golden hash. `result_of` computes the native per-op result field
/// (rw consumes it as `rd_val`). Returns the golden sha for logging.
#[allow(clippy::too_many_arguments)]
fn write_fullprogram_fixture(
    chip: &str,
    air: &str,
    out: &Path,
    elf_path: &str,
    inputs: &[BinaryInput],
    result_of: impl Fn(u8, u64, u64) -> Result<u64>,
    trace: &[u64],
    rows: usize,
    n_cols: usize,
    ops_seen: u64,
    frops: u64,
    instance: Option<(usize, usize)>,
    debug_trace: bool,
) -> Result<String> {
    let mut hasher = Sha256::new();
    for v in trace {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::with_capacity(inputs.len() * 32);
    for bi in inputs {
        let result = result_of(bi.op, bi.a, bi.b)?;
        rec_bytes.push(bi.op);
        rec_bytes.extend_from_slice(&[0u8; 7]);
        rec_bytes.extend_from_slice(&bi.a.to_le_bytes());
        rec_bytes.extend_from_slice(&bi.b.to_le_bytes());
        rec_bytes.extend_from_slice(&result.to_le_bytes());
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let mut meta = serde_json::json!({
        "chip": chip,
        "case": "fullprogram",
        "source_elf": elf_path,
        "air": air,
        "input_count": inputs.len(),
        "ops_seen": ops_seen,
        "frops_diverted": frops,
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { op:u8, pad[7], a:u64, b:u64, result:u64 }",
        "note": "FROPS-diverted ops are NOT in input_records (they emit rows in a separate AIR), so this trace needs no FROPS modeling on the rw side.",
    });
    if let Some((instance_id, total_instances)) = instance {
        meta["instance_id"] = serde_json::json!(instance_id);
        meta["total_instances"] = serde_json::json!(total_instances);
    }
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote {chip}/fullprogram: {} ({rows} rows x {n_cols} cols, {} inputs, {frops} FROPS)\n  golden_sha256: {golden_sha256}",
        out.display(),
        inputs.len(),
    );
    if debug_trace {
        let name = format!("expected_{chip}_trace.npy.gz");
        write_npy_gz(&out.join(&name), trace, rows, n_cols)?;
        println!("  + {name} (golden preimage, not committed)");
    }
    Ok(golden_sha256)
}

/// Common tail for every ArithEq curve full-program fixture: drive
/// `ArithEqSM::compute_witness` over the wrapped inputs, hash the trace into the
/// canonical-u64 golden, and write `arith_eq/<op>/fullprogram/{input_records.bin,
/// fixture_metadata.json}`. Only the record byte layout and the `ArithEqInput`
/// variant differ across the six curve ops, so the typed head (take the collected
/// inputs, serialize the wire records, wrap them) lives in `emit_arith_eq_add!` /
/// `emit_arith_eq_dbl!` and this tail is shared verbatim. `rec_bytes` is the
/// already-serialized record blob, so the typed inputs can move into `arith_eq_in`
/// first. ArithEq's `compute_witness` needs the `SetupCtx` (unlike the other
/// precompile SMs).
#[allow(clippy::too_many_arguments)]
fn write_arith_eq_fixture(
    out_base: &Path,
    op_name: &str,
    record_layout: &str,
    elf_path: &str,
    rec_bytes: Vec<u8>,
    input_count: usize,
    arith_eq_in: Vec<Vec<ArithEqInput>>,
    std: &Arc<Std<F>>,
    sctx: &Arc<SetupCtx<F>>,
    debug_trace: bool,
) -> Result<()> {
    let out = out_base.join(format!("arith_eq/{op_name}/fullprogram"));

    let arith_eq_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let arith_eq_sm = ArithEqSM::<F>::new(std.clone());
    let arith_eq_buf =
        vec![F::default(); arith_eq_rows * ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE];
    let arith_eq_air =
        arith_eq_sm.compute_witness::<ArithEqTraceRow<F>>(sctx, &arith_eq_in, arith_eq_buf)?;
    let arith_eq_cols = arith_eq_air.n_cols_trace;
    let rows = arith_eq_air.trace.len() / arith_eq_cols;
    let golden_sha256 = trace_golden_sha256(&arith_eq_air.trace);

    fs::create_dir_all(&out)?;
    fs::write(out.join("input_records.bin"), &rec_bytes)?;
    let meta = serde_json::json!({
        "chip": "arith_eq",
        "case": "fullprogram",
        "op": op_name,
        "source_elf": elf_path,
        "air": format!("ArithEq (air_id 26), op {op_name}"),
        "input_count": input_count,
        "ops_seen": input_count,
        "frops_diverted": 0,
        "trace_rows": rows,
        "trace_cols": arith_eq_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": record_layout,
        "rows_per_op": 16,
        "note": format!("Precompile — no FROPS diversion; every {op_name} op is one input."),
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote arith_eq/{op_name}/fullprogram: {} ({rows} rows x {arith_eq_cols} cols, {input_count} inputs, 0 FROPS)\n  golden_sha256: {golden_sha256}",
        out.display(),
    );
    if debug_trace {
        write_trace_dump(
            &out.join("expected_arith_eq_trace.npy.gz"),
            &arith_eq_air.trace,
            rows,
            arith_eq_cols,
        )?;
        println!("  + expected_arith_eq_trace.npy.gz (golden preimage, not committed)");
    }
    Ok(())
}

/// Emits one ArithEq curve **add** fixture (152-byte two-point record:
/// `step, addr, p1_addr, p2_addr, _pad, p1[8], p2[8]`, all LE). Takes the op's
/// collected inputs off `$bus.$field`, skips when the guest emitted none,
/// serializes the wire records in rw `ZiskArithEq<Curve>AddOpRecord` order, wraps
/// them as `ArithEqInput::$variant`, and defers the rest to
/// `write_arith_eq_fixture`.
macro_rules! emit_arith_eq_add {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 152);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.p1_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.p2_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&0u32.to_le_bytes());
                for x in r.p1.iter().chain(&r.p2) {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEqInput>> =
                vec![inputs.into_iter().map(ArithEqInput::$variant).collect()];
            write_arith_eq_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, p1_addr:u32, p2_addr:u32, _pad:u32, p1:[u64;8], p2:[u64;8] } = 152 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Emits one ArithEq curve **dbl** fixture (80-byte single-point record:
/// `step, addr, _pad, p1[8]`, all LE) — the doubling sibling of
/// `emit_arith_eq_add!`. Same flow with fewer record fields (no second point or
/// point addresses).
macro_rules! emit_arith_eq_dbl {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 80);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&0u32.to_le_bytes());
                for x in r.p1.iter() {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEqInput>> =
                vec![inputs.into_iter().map(ArithEqInput::$variant).collect()];
            write_arith_eq_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, _pad:u32, p1:[u64;8] } = 80 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Emits one ArithEq **complex** Fp2 fixture (152-byte two-Fp2-operand record:
/// `step, addr, f1_addr, f2_addr, _pad, f1[8], f2[8]`, all LE). The Fp2 sibling
/// of `emit_arith_eq_add!`: identical byte layout and flow, but the two operands
/// are field elements `f = (real ‖ imag)` rather than curve points, so the wire
/// records carry `f1_addr/f2_addr` + `f1/f2` (rw `ZiskArithEqBn254ComplexOpRecord`
/// order).
macro_rules! emit_arith_eq_complex {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 152);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.f1_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.f2_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&0u32.to_le_bytes());
                for x in r.f1.iter().chain(&r.f2) {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEqInput>> =
                vec![inputs.into_iter().map(ArithEqInput::$variant).collect()];
            write_arith_eq_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, f1_addr:u32, f2_addr:u32, _pad:u32, f1:[u64;8], f2:[u64;8] } = 152 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Emits one ArithEq **arith256** fixture (128-byte record:
/// `step, addr, a_addr, b_addr, c_addr, dl_addr, dh_addr, a[4], b[4], c[4]`, all
/// LE). The non-curve sibling of `emit_arith_eq_add!`: three 256-bit operands
/// (`a*b + c = dh:dl`) and no second point or `_pad`, so the wire records carry
/// the per-operand addresses + the two result-half addresses (rw
/// `ZiskArithEqArith256OpRecord` order).
macro_rules! emit_arith_eq_arith256 {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 128);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.a_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.b_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.c_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.dl_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.dh_addr.to_le_bytes());
                for x in r.a.iter().chain(&r.b).chain(&r.c) {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEqInput>> =
                vec![inputs.into_iter().map(ArithEqInput::$variant).collect()];
            write_arith_eq_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, a_addr:u32, b_addr:u32, c_addr:u32, dl_addr:u32, dh_addr:u32, a:[u64;4], b:[u64;4], c:[u64;4] } = 128 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Emits one ArithEq **arith256_mod** fixture (160-byte record:
/// `step, addr, a_addr, b_addr, c_addr, module_addr, d_addr, a[4], b[4], c[4],
/// module[4]`, all LE) — the modular sibling of `emit_arith_eq_arith256!`. Same
/// flow with one extra operand (`module`) and one result (`d = (a*b + c) mod
/// module`), so the record carries `module_addr`/`d_addr` + the `module` limbs
/// (rw `ZiskArithEqArith256ModOpRecord` order).
macro_rules! emit_arith_eq_arith256_mod {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 160);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.a_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.b_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.c_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.module_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.d_addr.to_le_bytes());
                for x in r.a.iter().chain(&r.b).chain(&r.c).chain(&r.module) {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEqInput>> =
                vec![inputs.into_iter().map(ArithEqInput::$variant).collect()];
            write_arith_eq_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, a_addr:u32, b_addr:u32, c_addr:u32, module_addr:u32, d_addr:u32, a:[u64;4], b:[u64;4], c:[u64;4], module:[u64;4] } = 160 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// The arith_eq_384 analog of `write_arith_eq_fixture`: drive
/// `ArithEq384SM::compute_witness` over the wrapped inputs on the *separate*
/// 384-bit air (air_id 27), hash the trace into the canonical-u64 golden, and
/// write `arith_eq_384/<op>/fullprogram/{input_records.bin, fixture_metadata.json}`.
/// Distinct SM / trace type from the 256-bit ArithEq, so it can't share that
/// tail; the typed head (serialize the wire records, wrap them) lives in the
/// `emit_arith_eq_384_*!` macros.
#[allow(clippy::too_many_arguments)]
fn write_arith_eq_384_fixture(
    out_base: &Path,
    op_name: &str,
    record_layout: &str,
    elf_path: &str,
    rec_bytes: Vec<u8>,
    input_count: usize,
    arith_eq_in: Vec<Vec<ArithEq384Input>>,
    std: &Arc<Std<F>>,
    sctx: &Arc<SetupCtx<F>>,
    debug_trace: bool,
) -> Result<()> {
    let out = out_base.join(format!("arith_eq_384/{op_name}/fullprogram"));

    let arith_eq_rows = ArithEq384Trace::<ArithEq384TraceRow<F>>::NUM_ROWS;
    let arith_eq_sm = ArithEq384SM::<F>::new(std.clone());
    let arith_eq_buf =
        vec![F::default(); arith_eq_rows * ArithEq384Trace::<ArithEq384TraceRow<F>>::ROW_SIZE];
    let arith_eq_air =
        arith_eq_sm.compute_witness::<ArithEq384TraceRow<F>>(sctx, &arith_eq_in, arith_eq_buf)?;
    let arith_eq_cols = arith_eq_air.n_cols_trace;
    let rows = arith_eq_air.trace.len() / arith_eq_cols;
    let golden_sha256 = trace_golden_sha256(&arith_eq_air.trace);

    fs::create_dir_all(&out)?;
    fs::write(out.join("input_records.bin"), &rec_bytes)?;
    let meta = serde_json::json!({
        "chip": "arith_eq_384",
        "case": "fullprogram",
        "op": op_name,
        "source_elf": elf_path,
        "air": format!("ArithEq384 (air_id 27), op {op_name}"),
        "input_count": input_count,
        "ops_seen": input_count,
        "frops_diverted": 0,
        "trace_rows": rows,
        "trace_cols": arith_eq_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": record_layout,
        "rows_per_op": 24,
        "note": format!("Precompile — no FROPS diversion; every {op_name} op is one input."),
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote arith_eq_384/{op_name}/fullprogram: {} ({rows} rows x {arith_eq_cols} cols, {input_count} inputs, 0 FROPS)\n  golden_sha256: {golden_sha256}",
        out.display(),
    );
    if debug_trace {
        write_trace_dump(
            &out.join("expected_arith_eq_384_trace.npy.gz"),
            &arith_eq_air.trace,
            rows,
            arith_eq_cols,
        )?;
        println!("  + expected_arith_eq_384_trace.npy.gz (golden preimage, not committed)");
    }
    Ok(())
}

/// Emits one ArithEq384 **arith384_mod** fixture (224-byte record:
/// `step, addr, a_addr, b_addr, c_addr, module_addr, d_addr, a[6], b[6], c[6],
/// module[6]`, all LE) — the 384-bit / 6-limb analog of
/// `emit_arith_eq_arith256_mod!`, on the separate arith_eq_384 air. Defers to
/// `write_arith_eq_384_fixture`.
macro_rules! emit_arith_eq_384_mod {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq_384/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 224);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.a_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.b_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.c_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.module_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.d_addr.to_le_bytes());
                for x in r.a.iter().chain(&r.b).chain(&r.c).chain(&r.module) {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEq384Input>> =
                vec![inputs.into_iter().map(ArithEq384Input::$variant).collect()];
            write_arith_eq_384_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, a_addr:u32, b_addr:u32, c_addr:u32, module_addr:u32, d_addr:u32, a:[u64;6], b:[u64;6], c:[u64;6], module:[u64;6] } = 224 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Emits one ArithEq384 bls12-381 **curve add** fixture (216-byte two-point
/// record: `step, addr, p1_addr, p2_addr, _pad, p1[12], p2[12]`, all LE) — the
/// 384-bit / 12-limb-point analog of `emit_arith_eq_add!`, on the arith_eq_384
/// air. Defers to `write_arith_eq_384_fixture`.
macro_rules! emit_arith_eq_384_curve_add {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq_384/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 216);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.p1_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.p2_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&0u32.to_le_bytes());
                for x in r.p1.iter().chain(&r.p2) {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEq384Input>> =
                vec![inputs.into_iter().map(ArithEq384Input::$variant).collect()];
            write_arith_eq_384_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, p1_addr:u32, p2_addr:u32, _pad:u32, p1:[u64;12], p2:[u64;12] } = 216 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Emits one ArithEq384 bls12-381 **curve dbl** fixture (112-byte single-point
/// record: `step, addr, _pad, p1[12]`, all LE) — the doubling sibling of
/// `emit_arith_eq_384_curve_add!`. Defers to `write_arith_eq_384_fixture`.
macro_rules! emit_arith_eq_384_curve_dbl {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq_384/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 112);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&0u32.to_le_bytes());
                for x in r.p1.iter() {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEq384Input>> =
                vec![inputs.into_iter().map(ArithEq384Input::$variant).collect()];
            write_arith_eq_384_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, _pad:u32, p1:[u64;12] } = 112 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Emits one ArithEq384 bls12-381 **complex** Fp2 fixture (216-byte
/// two-Fp2-operand record: `step, addr, f1_addr, f2_addr, _pad, f1[12], f2[12]`,
/// all LE). The Fp2 sibling of `emit_arith_eq_384_curve_add!`: identical byte
/// layout and flow, but the two operands are field elements `f = (real ‖ imag)`
/// rather than curve points (rw `ZiskArithEq384Bls12381ComplexOpRecord` order,
/// shared by add/sub/mul).
macro_rules! emit_arith_eq_384_complex {
    ($bus:ident, $field:ident, $op:literal, $variant:ident,
     $out_base:expr, $elf:expr, $std:expr, $sctx:expr, $debug_trace:expr) => {{
        let inputs = std::mem::take(&mut $bus.$field);
        if inputs.is_empty() {
            println!(concat!(
                "  note: no ",
                $op,
                " ops in this guest — skipping arith_eq_384/",
                $op
            ));
        } else {
            let input_count = inputs.len();
            let mut rec_bytes = Vec::with_capacity(input_count * 216);
            for r in &inputs {
                rec_bytes.extend_from_slice(&r.step.to_le_bytes());
                rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.f1_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&r.f2_addr.to_le_bytes());
                rec_bytes.extend_from_slice(&0u32.to_le_bytes());
                for x in r.f1.iter().chain(&r.f2) {
                    rec_bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            let arith_eq_in: Vec<Vec<ArithEq384Input>> =
                vec![inputs.into_iter().map(ArithEq384Input::$variant).collect()];
            write_arith_eq_384_fixture(
                $out_base,
                $op,
                "repr(C) { step:u64, addr:u32, f1_addr:u32, f2_addr:u32, _pad:u32, f1:[u64;12], f2:[u64;12] } = 216 bytes LE",
                $elf,
                rec_bytes,
                input_count,
                arith_eq_in,
                $std,
                $sctx,
                $debug_trace,
            )?;
        }
    }};
}

/// Gate-0 full-program path: transpile + emulate a guest ELF once, collect the
/// real run's BinaryBasic + BinaryExtension inputs off the operation bus (reusing
/// the native collectors), then drive each chip's `compute_witness` exactly as the
/// per-op fixture path does — emitting `binary/fullprogram` and
/// `binary_extension/fullprogram` fixtures under `out_base`. Reports each chip's
/// FROPS split so "every op" vs "what the AIR consumes" stays visible.
fn emit_fullprogram(
    elf_path: &str,
    inputs_path: Option<&str>,
    out_base: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let elf_bytes = fs::read(elf_path)?;
    let rom: ZiskRom = Riscv2zisk::new(&elf_bytes)
        .run()
        .map_err(|e| anyhow::anyhow!("riscv2zisk transpile failed: {e}"))?;

    let input_data: Vec<u8> = match inputs_path {
        Some(p) => fs::read(p)?,
        None => Vec::new(),
    };

    let emu_options = EmuOptions {
        chunk_size: Some(1 << 18),
        max_steps: 0xF_FFFF_FFFF,
        ..EmuOptions::default()
    };
    // Single thread ⇒ deterministic chunk order ⇒ deterministic input order.
    let min_traces = ZiskEmulator::compute_minimal_traces(&rom, &input_data, &emu_options, 1)?;

    // One AIR instance holds NUM_ROWS ops; size each collector to its chip's AIR
    // and warn if a run overflows it (multi-instance is out of Gate-0 scope).
    let basic_rows = BinaryTrace::<BinaryTraceRow<F>>::NUM_ROWS;
    let ext_rows = BinaryExtensionTrace::<BinaryExtensionTraceRow<F>>::NUM_ROWS;
    let mut bus = BinaryOracleBus {
        basic: BinaryBasicCollector::<F>::new(
            basic_rows,
            CollectSkipper::new(0),
            true, // with_adds: this single collector owns add + non-add basic ops
            true, // force_execute_to_end: drain every chunk
            std.clone(),
        ),
        extension: BinaryExtensionCollector::<F>::new(
            ext_rows,
            CollectSkipper::new(0),
            true, // force_execute_to_end
            std.clone(),
        ),
        arith: Vec::new(),
        arith_frops_table: std
            .get_virtual_table_id(ArithFrops::TABLE_ID)
            .expect("Failed to get Arith FROPS table ID"),
        sha256: Vec::new(),
        keccak: Vec::new(),
        secp256k1_add: Vec::new(),
        secp256k1_dbl: Vec::new(),
        secp256r1_add: Vec::new(),
        secp256r1_dbl: Vec::new(),
        bn254_curve_add: Vec::new(),
        bn254_curve_dbl: Vec::new(),
        bn254_complex_add: Vec::new(),
        bn254_complex_sub: Vec::new(),
        bn254_complex_mul: Vec::new(),
        arith256: Vec::new(),
        arith256_mod: Vec::new(),
        arith384_mod: Vec::new(),
        bls12_381_curve_add: Vec::new(),
        bls12_381_curve_dbl: Vec::new(),
        bls12_381_complex_add: Vec::new(),
        bls12_381_complex_sub: Vec::new(),
        bls12_381_complex_mul: Vec::new(),
        std: std.clone(),
        binary_ops_seen: 0,
        binary_e_ops_seen: 0,
        arith_ops_seen: 0,
    };
    for emu_trace in &min_traces {
        ZiskEmulator::process_emu_trace::<F, (), BinaryOracleBus>(&rom, emu_trace, &mut bus, false);
    }
    println!("full-program collect: {} chunks", min_traces.len());

    // BinaryBasic.
    let basic_inputs = std::mem::take(&mut bus.basic.inputs);
    let basic_frops = bus.binary_ops_seen.saturating_sub(basic_inputs.len() as u64);
    if basic_inputs.len() == basic_rows {
        println!("  WARNING: BinaryBasic hit NUM_ROWS={basic_rows} — run spans >1 instance");
    }
    let basic_sm = BinaryBasicSM::<F>::new(std.clone());
    let basic_buf =
        vec![F::default(); basic_rows * BinaryTrace::<BinaryTraceRow<F>>::ROW_SIZE];
    let basic_in = vec![basic_inputs];
    let basic_air = basic_sm.compute_witness::<BinaryTraceRow<F>>(&basic_in, basic_buf)?;
    let basic_cols = basic_air.n_cols_trace;
    let basic_trace: Vec<u64> = basic_air.trace.iter().map(|f| f.as_canonical_u64()).collect();
    write_fullprogram_fixture(
        "binary",
        "BinaryBasic (air_id 22)",
        &out_base.join("binary/fullprogram"),
        elf_path,
        &basic_in[0],
        binary_result,
        &basic_trace,
        basic_trace.len() / basic_cols,
        basic_cols,
        bus.binary_ops_seen,
        basic_frops,
        None,
        debug_trace,
    )?;

    // BinaryExtension.
    let ext_inputs = std::mem::take(&mut bus.extension.inputs);
    let ext_frops = bus.binary_e_ops_seen.saturating_sub(ext_inputs.len() as u64);
    if ext_inputs.len() == ext_rows {
        println!("  WARNING: BinaryExtension hit NUM_ROWS={ext_rows} — run spans >1 instance");
    }
    let ext_sm = BinaryExtensionSM::<F>::new(std.clone());
    let ext_buf = vec![
        F::default();
        ext_rows * BinaryExtensionTrace::<BinaryExtensionTraceRow<F>>::ROW_SIZE
    ];
    let ext_in = vec![ext_inputs];
    let ext_air = ext_sm.compute_witness::<BinaryExtensionTraceRow<F>>(&ext_in, ext_buf)?;
    let ext_cols = ext_air.n_cols_trace;
    let ext_trace: Vec<u64> = ext_air.trace.iter().map(|f| f.as_canonical_u64()).collect();
    write_fullprogram_fixture(
        "binary_extension",
        "BinaryExtension (air_id 24)",
        &out_base.join("binary_extension/fullprogram"),
        elf_path,
        &ext_in[0],
        binary_extension_result,
        &ext_trace,
        ext_trace.len() / ext_cols,
        ext_cols,
        bus.binary_e_ops_seen,
        ext_frops,
        None,
        debug_trace,
    )?;

    // Arith (air_id 21) — div/mul/rem. Inputs are raw bus OperationData; the AIR
    // recomputes the result from a,b, so the record's result field is unused.
    // Skipped when the guest emits none (ArithFullSM panics on empty inputs) —
    // e.g. a pure-Sha256f guest does no multiply/divide.
    let arith_inputs = std::mem::take(&mut bus.arith);
    if arith_inputs.is_empty() {
        println!("  note: no Arith ops in this guest — skipping arith/fullprogram");
    } else {
        let arith_rows = ArithTrace::<ArithTraceRow<F>>::NUM_ROWS;
        let arith_frops = bus.arith_ops_seen.saturating_sub(arith_inputs.len() as u64);
        if arith_inputs.len() == arith_rows {
            println!("  WARNING: Arith hit NUM_ROWS={arith_rows} — run spans >1 instance");
        }
        let arith_sm = ArithFullSM::<F>::new(std.clone());
        let arith_buf = vec![F::default(); arith_rows * ArithTrace::<ArithTraceRow<F>>::ROW_SIZE];
        let arith_in = vec![arith_inputs];
        let arith_air = arith_sm.compute_witness::<ArithTraceRow<F>>(&arith_in, arith_buf)?;
        let arith_cols = arith_air.n_cols_trace;
        let arith_trace: Vec<u64> =
            arith_air.trace.iter().map(|f| f.as_canonical_u64()).collect();
        let arith_bi: Vec<BinaryInput> =
            arith_in[0].iter().map(|d| BinaryInput::new(d[OP] as u8, d[A], d[B])).collect();
        write_fullprogram_fixture(
            "arith",
            "Arith (air_id 21)",
            &out_base.join("arith/fullprogram"),
            elf_path,
            &arith_bi,
            |_, _, _| Ok(0), // result unused by Arith (AIR recomputes from a,b)
            &arith_trace,
            arith_trace.len() / arith_cols,
            arith_cols,
            bus.arith_ops_seen,
            arith_frops,
            None,
            debug_trace,
        )?;
    }

    // Sha256f (air_id 29) — precompile, no FROPS (every op is an input). The
    // record layout matches the per-op sha256 fixture (116-byte packed), so rw
    // reads full-program and per-op records through the same reader. Skipped when
    // the guest emits no Sha256f ops (e.g. go_hello_world).
    let sha256_inputs = std::mem::take(&mut bus.sha256);
    let sha256_sm = Sha256fSM::<F>::new(std.clone());
    if sha256_inputs.is_empty() {
        println!("  note: no Sha256f ops in this guest — skipping sha256/fullprogram");
    } else if sha256_inputs.len() > sha256_sm.num_available_sha256fs {
        // compute_witness panics past one instance; multi-instance is out of
        // Gate-0 scope, so skip rather than abort the whole run.
        println!(
            "  note: {} Sha256f ops exceed one instance ({} available) — skipping sha256/fullprogram (multi-instance out of Gate-0 scope)",
            sha256_inputs.len(),
            sha256_sm.num_available_sha256fs,
        );
    } else {
        let sha256_out = out_base.join("sha256/fullprogram");
        let input_count = sha256_inputs.len();

        // Serialize the wire records before the inputs move into the SM.
        let mut rec_bytes = Vec::with_capacity(input_count * 116);
        for r in &sha256_inputs {
            rec_bytes.extend_from_slice(&r.step_main.to_le_bytes());
            rec_bytes.extend_from_slice(&r.addr_main.to_le_bytes());
            rec_bytes.extend_from_slice(&r.state_addr.to_le_bytes());
            rec_bytes.extend_from_slice(&r.input_addr.to_le_bytes());
            for x in r.state.iter().chain(&r.input) {
                rec_bytes.extend_from_slice(&x.to_le_bytes());
            }
        }

        let sha256_rows = Sha256fTrace::<Sha256fTraceRow<F>>::NUM_ROWS;
        let sha256_buf =
            vec![F::default(); sha256_rows * Sha256fTrace::<Sha256fTraceRow<F>>::ROW_SIZE];
        let sha256_in = vec![sha256_inputs];
        let sha256_air =
            sha256_sm.compute_witness::<Sha256fTraceRow<F>>(&sctx, &sha256_in, sha256_buf)?;
        let sha256_cols = sha256_air.n_cols_trace;
        let rows = sha256_air.trace.len() / sha256_cols;
        let golden_sha256 = trace_golden_sha256(&sha256_air.trace);

        fs::create_dir_all(&sha256_out)?;
        fs::write(sha256_out.join("input_records.bin"), &rec_bytes)?;
        let meta = serde_json::json!({
            "chip": "sha256",
            "case": "fullprogram",
            "source_elf": elf_path,
            "air": "Sha256f (air_id 29)",
            "input_count": input_count,
            "ops_seen": input_count,
            "frops_diverted": 0,
            "trace_rows": rows,
            "trace_cols": sha256_cols,
            "field": "goldilocks_canonical_u64",
            "golden_sha256": golden_sha256,
            "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
            "record_layout": "repr(C) packed { step_main:u64, addr_main:u32, state_addr:u32, input_addr:u32, state:[u64;4], input:[u64;8] } = 116 bytes LE",
            "note": "Precompile — no FROPS diversion; every Sha256f op is one input.",
        });
        fs::write(
            sha256_out.join("fixture_metadata.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;
        println!(
            "wrote sha256/fullprogram: {} ({rows} rows x {sha256_cols} cols, {input_count} inputs, 0 FROPS)\n  golden_sha256: {golden_sha256}",
            sha256_out.display(),
        );
        if debug_trace {
            write_trace_dump(
                &sha256_out.join("expected_sha256_trace.npy.gz"),
                &sha256_air.trace,
                rows,
                sha256_cols,
            )?;
            println!("  + expected_sha256_trace.npy.gz (golden preimage, not committed)");
        }
    }

    // Keccakf (air_id 28) — precompile, no FROPS (every op is an input). The
    // record layout matches the per-op keccak fixture (212-byte packed), so rw
    // reads full-program and per-op records through the same reader. Skipped when
    // the guest emits no Keccakf ops (e.g. go_hello_world).
    let keccak_inputs = std::mem::take(&mut bus.keccak);
    let keccak_sm = KeccakfSM::<F>::new(std.clone());
    if keccak_inputs.is_empty() {
        println!("  note: no Keccakf ops in this guest — skipping keccak/fullprogram");
    } else if keccak_inputs.len() > keccak_sm.num_available_keccakfs {
        // compute_witness panics past one instance (zec-reth on block 21740136
        // carries 22919 Keccakf perms vs 5241 per instance); multi-instance is
        // out of Gate-0 scope, so skip rather than abort the whole run.
        println!(
            "  note: {} Keccakf ops exceed one instance ({} available) — skipping keccak/fullprogram (multi-instance out of Gate-0 scope)",
            keccak_inputs.len(),
            keccak_sm.num_available_keccakfs,
        );
    } else {
        let keccak_out = out_base.join("keccak/fullprogram");
        let input_count = keccak_inputs.len();

        // Serialize the wire records before the inputs move into the SM.
        let mut rec_bytes = Vec::with_capacity(input_count * 212);
        for r in &keccak_inputs {
            rec_bytes.extend_from_slice(&r.step_main.to_le_bytes());
            rec_bytes.extend_from_slice(&r.addr_main.to_le_bytes());
            for lane in &r.state {
                rec_bytes.extend_from_slice(&lane.to_le_bytes());
            }
        }

        let keccak_rows = KeccakfTrace::<KeccakfTraceRow<F>>::NUM_ROWS;
        let keccak_buf =
            vec![F::default(); keccak_rows * KeccakfTrace::<KeccakfTraceRow<F>>::ROW_SIZE];
        let keccak_in = vec![keccak_inputs];
        let keccak_air =
            keccak_sm.compute_witness::<KeccakfTraceRow<F>>(&sctx, &keccak_in, keccak_buf)?;
        let keccak_cols = keccak_air.n_cols_trace;
        let rows = keccak_air.trace.len() / keccak_cols;
        let golden_sha256 = trace_golden_sha256(&keccak_air.trace);

        fs::create_dir_all(&keccak_out)?;
        fs::write(keccak_out.join("input_records.bin"), &rec_bytes)?;
        let meta = serde_json::json!({
            "chip": "keccak",
            "case": "fullprogram",
            "source_elf": elf_path,
            "air": "Keccakf (air_id 28)",
            "input_count": input_count,
            "ops_seen": input_count,
            "frops_diverted": 0,
            "trace_rows": rows,
            "trace_cols": keccak_cols,
            "field": "goldilocks_canonical_u64",
            "golden_sha256": golden_sha256,
            "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
            "record_layout": "repr(C) packed { step_main:u64, addr_main:u32, state:[u64;25] } = 212 bytes LE",
            "note": "Precompile — no FROPS diversion; every Keccakf op is one input.",
        });
        fs::write(
            keccak_out.join("fixture_metadata.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;
        println!(
            "wrote keccak/fullprogram: {} ({rows} rows x {keccak_cols} cols, {input_count} inputs, 0 FROPS)\n  golden_sha256: {golden_sha256}",
            keccak_out.display(),
        );
        if debug_trace {
            write_trace_dump(
                &keccak_out.join("expected_keccak_trace.npy.gz"),
                &keccak_air.trace,
                rows,
                keccak_cols,
            )?;
            println!("  + expected_keccak_trace.npy.gz (golden preimage, not committed)");
        }
    }

    // ArithEq ops (air_id 26) — precompile, no FROPS (every op is one input).
    // Each single-op guest emits only its own variant; emit_arith_eq_add! /
    // emit_arith_eq_dbl! / emit_arith_eq_complex! serialize the wire records
    // (152-byte two-point / 80-byte single-point / 152-byte two-Fp2-operand) and
    // drive ArithEqSM via write_arith_eq_fixture, which needs the SetupCtx (unlike
    // the other precompile SMs). Skipped per op when the guest emits none (e.g.
    // go_hello_world).
    emit_arith_eq_add!(
        bus, secp256k1_add, "secp256k1_add", Secp256k1Add, out_base, elf_path, &std, &sctx, debug_trace
    );

    emit_arith_eq_dbl!(
        bus, secp256k1_dbl, "secp256k1_dbl", Secp256k1Dbl, out_base, elf_path, &std, &sctx, debug_trace
    );

    emit_arith_eq_add!(
        bus, secp256r1_add, "secp256r1_add", Secp256r1Add, out_base, elf_path, &std, &sctx, debug_trace
    );

    emit_arith_eq_dbl!(
        bus, secp256r1_dbl, "secp256r1_dbl", Secp256r1Dbl, out_base, elf_path, &std, &sctx, debug_trace
    );

    emit_arith_eq_add!(
        bus, bn254_curve_add, "bn254_curve_add", Bn254CurveAdd, out_base, elf_path, &std, &sctx, debug_trace
    );

    emit_arith_eq_dbl!(
        bus, bn254_curve_dbl, "bn254_curve_dbl", Bn254CurveDbl, out_base, elf_path, &std, &sctx, debug_trace
    );

    emit_arith_eq_complex!(
        bus,
        bn254_complex_add,
        "bn254_complex_add",
        Bn254ComplexAdd,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );

    emit_arith_eq_complex!(
        bus,
        bn254_complex_sub,
        "bn254_complex_sub",
        Bn254ComplexSub,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );

    emit_arith_eq_complex!(
        bus,
        bn254_complex_mul,
        "bn254_complex_mul",
        Bn254ComplexMul,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );

    emit_arith_eq_arith256!(
        bus,
        arith256,
        "arith256",
        Arith256,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );
    emit_arith_eq_arith256_mod!(
        bus,
        arith256_mod,
        "arith256_mod",
        Arith256Mod,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );

    emit_arith_eq_384_mod!(
        bus,
        arith384_mod,
        "arith384_mod",
        Arith384Mod,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );
    emit_arith_eq_384_curve_add!(
        bus,
        bls12_381_curve_add,
        "bls12_381_curve_add",
        Bls12_381CurveAdd,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );
    emit_arith_eq_384_curve_dbl!(
        bus,
        bls12_381_curve_dbl,
        "bls12_381_curve_dbl",
        Bls12_381CurveDbl,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );
    emit_arith_eq_384_complex!(
        bus,
        bls12_381_complex_add,
        "bls12_381_complex_add",
        Bls12_381ComplexAdd,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );
    emit_arith_eq_384_complex!(
        bus,
        bls12_381_complex_sub,
        "bls12_381_complex_sub",
        Bls12_381ComplexSub,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );
    emit_arith_eq_384_complex!(
        bus,
        bls12_381_complex_mul,
        "bls12_381_complex_mul",
        Bls12_381ComplexMul,
        out_base,
        elf_path,
        &std,
        &sctx,
        debug_trace
    );

    Ok(())
}

/// Multi-instance binary-family emission (#2347): plan-driven, unlike the
/// single-instance Gate-0 path in `emit_fullprogram`. Runs the native
/// count→plan→expand pipeline — per-chunk `BinaryCounter` metrics, the real
/// `BinaryPlanner` (including its global `enable_bin_add_sm` cost decision that
/// may split adds onto the dedicated BinaryAdd air), then per plan the concrete
/// instance's plan-parameterized collectors ((count, force_execute_to_end,
/// CollectSkipper) per chunk) replayed in sorted chunk order — so each emitted
/// `instNN/` fixture matches a native AIR instance byte-for-byte. Emits every
/// planned instance of BinaryBasic / BinaryAdd / BinaryExtension under
/// `binary{,_add,_extension}/fullprogram/instNN/`.
macro_rules! emit_binary_air {
    ($plans:expr, $sm:expr, $instance:ident, $build:ident, $trace:ident, $row:ident,
     $chip:literal, $air_label:expr, $result_of:expr, $to_bi:expr, $rom:expr,
     $min_traces:expr, $out_base:expr, $elf_path:expr, $std:expr, $sctx:expr, $pctx:expr,
     $feed:ident, $debug_trace:expr) => {{
        let total = $plans.len();
        for (idx, plan) in $plans.into_iter().enumerate() {
            let chunk_ids = plan_chunk_ids(&plan.check_point, $chip)?;
            let instance = $instance::new($sm.clone(), InstanceCtx::new(0, plan), $std.clone());
            let mut collectors: Vec<(usize, Box<dyn BusDevice<u64>>)> = Vec::new();
            let mut serial: Vec<BinaryInput> = Vec::new();
            let mut ops_seen = 0u64;
            for cid in &chunk_ids {
                let collector = instance.$build(*cid);
                let mut bus = $feed { collector };
                ZiskEmulator::process_emu_trace::<F, (), $feed>(
                    $rom,
                    &$min_traces[cid.0],
                    &mut bus,
                    true,
                );
                let collector = bus.collector;
                ops_seen += collector.inputs.len() as u64;
                serial.extend(collector.inputs.iter().map($to_bi));
                collectors.push((cid.0, Box::new(collector) as Box<dyn BusDevice<u64>>));
            }
            let num_rows = $trace::<$row<F>>::NUM_ROWS;
            let buffer = vec![F::default(); num_rows * $trace::<$row<F>>::ROW_SIZE];
            let air = instance
                .compute_witness(&$pctx, &$sctx, collectors, buffer, false)?
                .ok_or_else(|| anyhow::anyhow!("{} instance produced no AirInstance", $chip))?;
            let n_cols = air.n_cols_trace;
            let trace_u64: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();
            write_fullprogram_fixture(
                $chip,
                &$air_label,
                &$out_base.join(format!("{}/fullprogram/inst{idx:02}", $chip)),
                $elf_path,
                &serial,
                $result_of,
                &trace_u64,
                trace_u64.len() / n_cols,
                n_cols,
                ops_seen,
                0, // per-instance FROPS split is planner-internal; inputs are exact
                Some((idx, total)),
                $debug_trace,
            )?;
        }
    }};
}

#[allow(clippy::too_many_arguments)]
fn emit_binary_multi(
    elf_path: &str,
    inputs_path: Option<&str>,
    out_base: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
    pctx: Arc<ProofCtx<F>>,
) -> Result<()> {
    let elf_bytes = fs::read(elf_path)?;
    let rom: ZiskRom = Riscv2zisk::new(&elf_bytes)
        .run()
        .map_err(|e| anyhow::anyhow!("riscv2zisk transpile failed: {e}"))?;
    let input_data: Vec<u8> = match inputs_path {
        Some(p) => fs::read(p)?,
        None => Vec::new(),
    };
    let emu_options = EmuOptions {
        chunk_size: Some(1 << 18),
        max_steps: 0xF_FFFF_FFFF,
        ..EmuOptions::default()
    };
    // Single thread ⇒ deterministic chunk order ⇒ deterministic input order.
    let min_traces = ZiskEmulator::compute_minimal_traces(&rom, &input_data, &emu_options, 1)?;

    // Count phase: per-chunk BinaryCounter metrics, as the executor collects them.
    let mut metrics: Vec<(ChunkId, Box<dyn BusDeviceMetrics>)> = Vec::new();
    for (i, emu_trace) in min_traces.iter().enumerate() {
        let mut bus = BinaryCountBus { counter: BinaryCounter::new() };
        ZiskEmulator::process_emu_trace::<F, (), BinaryCountBus>(&rom, emu_trace, &mut bus, true);
        metrics.push((ChunkId(i), Box::new(bus.counter)));
    }

    // Plan phase: the production planner, via the public ComponentPlanBuilder
    // surface (BinaryPlanner itself is crate-private).
    let plans = <BinarySM<F> as ComponentPlanBuilder<F>>::planner(false).plan(metrics);
    let mut basic_plans = Vec::new();
    let mut add_plans = Vec::new();
    let mut ext_plans = Vec::new();
    for plan in plans {
        if plan.instance_type != InstanceType::Instance {
            continue;
        }
        match plan.air_id {
            id if id == BinaryTrace::<()>::AIR_ID => basic_plans.push(plan),
            id if id == BinaryAddTrace::<()>::AIR_ID => add_plans.push(plan),
            id if id == BinaryExtensionTrace::<()>::AIR_ID => ext_plans.push(plan),
            _ => {}
        }
    }
    println!(
        "binary plan: {} BinaryBasic + {} BinaryAdd + {} BinaryExtension instance(s)",
        basic_plans.len(),
        add_plans.len(),
        ext_plans.len(),
    );

    let basic_sm = BinaryBasicSM::<F>::new(std.clone());
    emit_binary_air!(
        basic_plans,
        basic_sm,
        BinaryBasicInstance,
        build_binary_basic_collector,
        BinaryTrace,
        BinaryTraceRow,
        "binary",
        format!("BinaryBasic (air_id {})", BinaryTrace::<()>::AIR_ID),
        binary_result,
        |bi: &BinaryInput| BinaryInput::new(bi.op, bi.a, bi.b),
        &rom,
        min_traces,
        out_base,
        elf_path,
        std,
        sctx,
        pctx,
        BinaryBasicFeedBus,
        debug_trace
    );

    let add_sm = BinaryAddSM::<F>::new(std.clone());
    emit_binary_air!(
        add_plans,
        add_sm,
        BinaryAddInstance,
        build_binary_add_collector,
        BinaryAddTrace,
        BinaryAddTraceRow,
        "binary_add",
        format!("BinaryAdd (air_id {})", BinaryAddTrace::<()>::AIR_ID),
        binary_result,
        // BinaryAdd's collector stores bare (a, b) pairs — the air has no opcode
        // column. Record them as Add ops so the shared 32-byte layout holds.
        |p: &[u64; 2]| BinaryInput::new(ADD_OP, p[0], p[1]),
        &rom,
        min_traces,
        out_base,
        elf_path,
        std,
        sctx,
        pctx,
        BinaryAddFeedBus,
        debug_trace
    );

    let ext_sm = BinaryExtensionSM::<F>::new(std.clone());
    emit_binary_air!(
        ext_plans,
        ext_sm,
        BinaryExtensionInstance,
        build_binary_extension_collector,
        BinaryExtensionTrace,
        BinaryExtensionTraceRow,
        "binary_extension",
        format!("BinaryExtension (air_id {})", BinaryExtensionTrace::<()>::AIR_ID),
        binary_extension_result,
        |bi: &BinaryInput| BinaryInput::new(bi.op, bi.a, bi.b),
        &rom,
        min_traces,
        out_base,
        elf_path,
        std,
        sctx,
        pctx,
        BinaryExtensionFeedBus,
        debug_trace
    );

    Ok(())
}

/// Multi-instance Keccakf emission (#2347): plan-driven, replacing the
/// single-instance capacity skip in `emit_fullprogram` for runs whose perm
/// count exceeds `num_available_keccakfs` (block 21740136: 22919 perms vs 5241
/// per 131072-row instance). Reproduces the native partition with the same
/// `zisk_common::plan` helper the macro-generated KeccakfPlanner calls, over
/// the same per-chunk op counts — sequential (skip, count) windows across the
/// chunk-ordered op stream — then drives `KeccakfSM::compute_witness` once per
/// window and emits `keccak/fullprogram/instNN/` fixtures.
fn emit_keccak_multi(
    elf_path: &str,
    inputs_path: Option<&str>,
    out_base: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let elf_bytes = fs::read(elf_path)?;
    let rom: ZiskRom = Riscv2zisk::new(&elf_bytes)
        .run()
        .map_err(|e| anyhow::anyhow!("riscv2zisk transpile failed: {e}"))?;
    let input_data: Vec<u8> = match inputs_path {
        Some(p) => fs::read(p)?,
        None => Vec::new(),
    };
    let emu_options = EmuOptions {
        chunk_size: Some(1 << 18),
        max_steps: 0xF_FFFF_FFFF,
        ..EmuOptions::default()
    };
    // Single thread ⇒ deterministic chunk order ⇒ deterministic input order.
    let min_traces = ZiskEmulator::compute_minimal_traces(&rom, &input_data, &emu_options, 1)?;

    // Capture every chunk's Keccakf stream once; counts feed the planner and
    // the streams get sliced per instance window.
    let mut streams: Vec<Vec<KeccakfInput>> = Vec::with_capacity(min_traces.len());
    for emu_trace in &min_traces {
        let mut bus = KeccakCaptureBus { inputs: Vec::new() };
        ZiskEmulator::process_emu_trace::<F, (), KeccakCaptureBus>(&rom, emu_trace, &mut bus, true);
        streams.push(bus.inputs);
    }
    let total_ops: usize = streams.iter().map(|s| s.len()).sum();
    if total_ops == 0 {
        bail!("no Keccakf ops in this guest — nothing to emit");
    }

    let keccak_sm = KeccakfSM::<F>::new(std.clone());
    let counts: Vec<InstCount> = streams
        .iter()
        .enumerate()
        .map(|(i, s)| InstCount::new(ChunkId(i), s.len() as u64))
        .collect();
    let windows = plan_instance_windows(&counts, keccak_sm.num_available_keccakfs as u64);
    let total_instances = windows.len();
    println!(
        "keccak plan: {total_ops} perms -> {total_instances} instance(s) of {}",
        keccak_sm.num_available_keccakfs,
    );

    for (idx, (check_point, collect_info)) in windows.into_iter().enumerate() {
        let chunk_ids = plan_chunk_ids(&check_point, "Keccakf")?;
        let mut inputs: Vec<KeccakfInput> = Vec::new();
        for cid in &chunk_ids {
            let (num_ops, skipper) = collect_info[cid];
            let start = skipper.skip as usize;
            let end = start + num_ops as usize;
            // KeccakfInput derives no Clone; its fields are all plain values.
            inputs.extend(streams[cid.0][start..end].iter().map(|r| KeccakfInput {
                step_main: r.step_main,
                addr_main: r.addr_main,
                state: r.state,
            }));
        }
        let input_count = inputs.len();

        // Serialize the wire records before the inputs move into the SM (same
        // 212-byte packed layout as the single-instance keccak fixture).
        let mut rec_bytes = Vec::with_capacity(input_count * 212);
        for r in &inputs {
            rec_bytes.extend_from_slice(&r.step_main.to_le_bytes());
            rec_bytes.extend_from_slice(&r.addr_main.to_le_bytes());
            for lane in &r.state {
                rec_bytes.extend_from_slice(&lane.to_le_bytes());
            }
        }

        let keccak_rows = KeccakfTrace::<KeccakfTraceRow<F>>::NUM_ROWS;
        let keccak_buf =
            vec![F::default(); keccak_rows * KeccakfTrace::<KeccakfTraceRow<F>>::ROW_SIZE];
        let keccak_air =
            keccak_sm.compute_witness::<KeccakfTraceRow<F>>(&sctx, &[inputs], keccak_buf)?;
        let keccak_cols = keccak_air.n_cols_trace;
        let rows = keccak_air.trace.len() / keccak_cols;
        let data: Vec<u64> = keccak_air.trace.iter().map(|f| f.as_canonical_u64()).collect();
        let mut hasher = Sha256::new();
        for v in &data {
            hasher.update(v.to_le_bytes());
        }
        let golden_sha256 =
            hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

        let inst_out = out_base.join(format!("keccak/fullprogram/inst{idx:02}"));
        fs::create_dir_all(&inst_out)?;
        fs::write(inst_out.join("input_records.bin"), &rec_bytes)?;
        let meta = serde_json::json!({
            "chip": "keccak",
            "case": "fullprogram",
            "source_elf": elf_path,
            "air": format!("Keccakf (air_id {})", KeccakfTrace::<()>::AIR_ID),
            "instance_id": idx,
            "total_instances": total_instances,
            "input_count": input_count,
            "ops_seen": input_count,
            "frops_diverted": 0,
            "trace_rows": rows,
            "trace_cols": keccak_cols,
            "field": "goldilocks_canonical_u64",
            "golden_sha256": golden_sha256,
            "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
            "record_layout": "repr(C) packed { step_main:u64, addr_main:u32, state:[u64;25] } = 212 bytes LE",
            "note": "Precompile — no FROPS diversion; every Keccakf op is one input.",
        });
        fs::write(inst_out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
        println!(
            "wrote keccak/fullprogram inst {idx}/{total_instances}: {} ({rows} rows x {keccak_cols} cols, {input_count} inputs)\n  golden_sha256: {golden_sha256}",
            inst_out.display(),
        );
        if debug_trace {
            write_trace_dump(
                &inst_out.join("expected_keccak_trace.npy.gz"),
                &keccak_air.trace,
                rows,
                keccak_cols,
            )?;
            println!("  + expected_keccak_trace.npy.gz (golden preimage, not committed)");
        }
    }

    Ok(())
}

/// Main-SM per-segment goldens (#2347 follow-on: the milestone's main-SM
/// gate). Drives the native `MainPlanner` + `MainInstance::compute_witness`
/// per segment (each segment = NUM_ROWS/chunk_size minimal traces, one row per
/// step) and emits `main/fullprogram/segNN/fixture_metadata.json` — golden
/// only, NO input_records.bin: the rw gate builds its own `ZiskExtStepData`
/// stream from the committed ZROM/ZPIN via `ZiskCheckpointReplay` +
/// `PopulateZiskMainDecodeFields` + `ZiskMainRegAccess`, so the gate covers
/// the rw step-producer and filler together against the native trace.
fn emit_main_multi(
    elf_path: &str,
    inputs_path: Option<&str>,
    out_base: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    // Name map for the flat airvalue vector: one slot per stage-1 entry,
    // three (extension) otherwise — the same sizing proofman's
    // initialize_air_instance uses.
    let av_map: Vec<(String, u64)> = sctx
        .get_setup(ZISK_AIRGROUP_ID, MainTrace::<()>::AIR_ID)
        .map_err(|e| anyhow::anyhow!("get_setup(Main): {e:?}"))?
        .stark_info
        .airvalues_map
        .as_ref()
        .map(|m| m.iter().map(|e| (e.name.clone(), e.stage)).collect())
        .unwrap_or_default();
    let elf_bytes = fs::read(elf_path)?;
    let rom: ZiskRom = Riscv2zisk::new(&elf_bytes)
        .run()
        .map_err(|e| anyhow::anyhow!("riscv2zisk transpile failed: {e}"))?;
    let input_data: Vec<u8> = match inputs_path {
        Some(p) => fs::read(p)?,
        None => Vec::new(),
    };
    let chunk_size: u64 = 1 << 18;
    let emu_options = EmuOptions {
        chunk_size: Some(chunk_size),
        max_steps: 0xF_FFFF_FFFF,
        ..EmuOptions::default()
    };
    // Single thread ⇒ deterministic chunk order ⇒ deterministic step order.
    let min_traces = ZiskEmulator::compute_minimal_traces(&rom, &input_data, &emu_options, 1)?;

    let mut plans = MainPlanner::plan(&min_traces, chunk_size)
        .map_err(|e| anyhow::anyhow!("MainPlanner::plan failed: {e:?}"))?;
    plans.sort_by_key(|p| p.segment_id.map(|s| s.0).unwrap_or(0));
    let total_segments = plans.len();
    println!(
        "main plan: {} chunks -> {total_segments} segment(s) of {} rows",
        min_traces.len(),
        MainTrace::<()>::NUM_ROWS,
    );

    for plan in plans {
        let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
        let instance = MainInstance::<F>::new(InstanceCtx::new(0, plan), std.clone());
        let num_rows = MainTrace::<MainTraceRow<F>>::NUM_ROWS;
        let buffer = vec![F::default(); num_rows * MainTrace::<MainTraceRow<F>>::ROW_SIZE];
        let air = instance
            .compute_witness::<MainTraceRow<F>>(&rom, &min_traces, chunk_size, buffer)
            .map_err(|e| anyhow::anyhow!("Main segment {seg} compute_witness failed: {e:?}"))?;
        let n_cols = air.n_cols_trace;
        let rows = air.trace.len() / n_cols;
        let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();
        let mut hasher = Sha256::new();
        for v in &data {
            hasher.update(v.to_le_bytes());
        }
        let golden_sha256 =
            hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();
        // The segment's air values in pilout order (MainAirValues layout:
        // main_segment, main_last_segment, segment pc/c carries, then the
        // per-register last_reg_value limb pairs + last_reg_mem_step) — the
        // cross-segment continuation state the rw side must export and the
        // constraint oracle must bind (riscv-witness#2356).
        let airvalues: Vec<u64> = air.airvalues.iter().map(|f| f.as_canonical_u64()).collect();
        let mut av_off = 0usize;
        let airvalues_named: Vec<serde_json::Value> = av_map
            .iter()
            .map(|(name, stage)| {
                let width = if *stage == 1 { 1 } else { 3 };
                let vals = &airvalues[av_off..av_off + width];
                av_off += width;
                serde_json::json!({"name": name, "stage": stage, "values": vals})
            })
            .collect();
        anyhow::ensure!(
            av_off == airvalues.len(),
            "airValuesMap width sum {av_off} != airvalues len {}",
            airvalues.len()
        );

        let seg_out = out_base.join(format!("main/fullprogram/seg{seg:02}"));
        fs::create_dir_all(&seg_out)?;
        let meta = serde_json::json!({
            "chip": "main",
            "case": "fullprogram",
            "source_elf": elf_path,
            "air": format!("Main (air_id {})", MainTrace::<()>::AIR_ID),
            "segment_id": seg,
            "total_segments": total_segments,
            "trace_rows": rows,
            "trace_cols": n_cols,
            "field": "goldilocks_canonical_u64",
            "golden_sha256": golden_sha256,
            "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
            "records_note": "no input_records.bin — the rw gate replays the committed ZROM/ZPIN through its own step producer (ZiskCheckpointReplay + step extension) and compares the filler output against this golden",
            "airvalues": airvalues,
            "airvalues_note": "canonical u64, flat in the pilout airValuesMap order (Main airgroup); names resolve via the proving key's pilout.globalInfo airValuesMap",
            "airvalues_named": airvalues_named,
        });
        fs::write(seg_out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
        println!(
            "wrote main/fullprogram seg {seg}/{total_segments}: {} ({rows} rows x {n_cols} cols)\n  golden_sha256: {golden_sha256}",
            seg_out.display(),
        );
        if debug_trace {
            write_trace_dump(&seg_out.join("expected_main_trace.npy.gz"), &air.trace, rows, n_cols)?;
            println!("  + expected_main_trace.npy.gz (golden preimage, not committed)");
        }
    }

    Ok(())
}

/// Slice of the segmented-mem oracle (#1845): emit the MemAlign (air_id 17)
/// full-program fixture. Same count→plan→expand shape as the Mem module, but the
/// MemAlign sub-machine carries no previous-segment, and its inputs are
/// `MemAlignInput` (width + the two aligned words read) serialized into the
/// 48-byte `repr(C)` record rw's `HashZiskMemAlignTrace` consumes.
#[allow(clippy::too_many_arguments)]
fn emit_mem_align(
    elf_path: &str,
    inputs_path: Option<&str>,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
    pctx: Arc<ProofCtx<F>>,
) -> Result<()> {
    let (rom, min_traces) = load_guest_min_traces(elf_path, inputs_path)?;

    let mut metrics: Vec<(ChunkId, Box<dyn BusDeviceMetrics>)> = Vec::new();
    for (i, emu_trace) in min_traces.iter().enumerate() {
        let mut bus = MemCountBus { counters: MemCounters::new(), mem_writes_seen: 0 };
        // The executor seeds the FIRST chunk's counter with the memory-init
        // sections before busing it (execution/rust.rs, `is_first()`); the
        // planner's offsets under-allocate every initialized address without
        // this, and the trace rows for the init writes land as overwrites.
        if i == 0 {
            bus.counters.init_with_mem_sections(&rom as &dyn zisk_core::MemDataSection);
        }
        ZiskEmulator::process_emu_trace::<F, (), MemCountBus>(&rom, emu_trace, &mut bus, true);
        bus.counters.close();
        metrics.push((ChunkId(i), Box::new(bus.counters)));
    }
    let plan = MemPlanner::new()
        .plan(metrics)
        .into_iter()
        .find(|p| p.air_id == MEM_ALIGN_AIR_IDS[0])
        .ok_or_else(|| anyhow::anyhow!("planner produced no MemAlign (air_id 17) plan"))?;
    if plan.segment_id != Some(zisk_common::SegmentId(0)) {
        bail!("expected a single MemAlign segment; got {:?}", plan.segment_id);
    }
    let chunk_ids = plan_chunk_ids(&plan.check_point, "memalign")?;

    let instance = MemAlignInstance::new(MemAlignSM::new(std), InstanceCtx::new(0, plan));
    let mut collectors: Vec<(usize, Box<dyn BusDevice<u64>>)> = Vec::new();
    let mut serial: Vec<MemAlignInput> = Vec::new();
    for cid in &chunk_ids {
        let collector = instance.build_mem_align_collector(*cid);
        let mut bus = MemAlignFeedBus { collector };
        ZiskEmulator::process_emu_trace::<F, (), MemAlignFeedBus>(
            &rom,
            &min_traces[cid.0],
            &mut bus,
            true,
        );
        let collector = bus.collector;
        for mi in &collector.inputs {
            serial.push(MemAlignInput {
                addr: mi.addr,
                is_write: mi.is_write,
                width: mi.width,
                step: mi.step,
                value: mi.value,
                mem_values: mi.mem_values,
            });
        }
        collectors.push((cid.0, Box::new(collector) as Box<dyn BusDevice<u64>>));
    }

    let num_rows = MemAlignTrace::<MemAlignTraceRow<F>>::NUM_ROWS;
    let buffer = vec![F::default(); num_rows * MemAlignTrace::<MemAlignTraceRow<F>>::ROW_SIZE];
    let air = instance
        .compute_witness(&pctx, &sctx, collectors, buffer, false)?
        .ok_or_else(|| anyhow::anyhow!("MemAlign instance produced no AirInstance"))?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let golden_sha256 = trace_golden_sha256(&air.trace);

    // Fixture: repr(C) { addr:u32, is_write:u32, width:u32, pad:u32, step:u64,
    // value:u64, mem_values:[u64;2] } (48 bytes).
    fs::create_dir_all(out)?;
    let mut rec = Vec::with_capacity(serial.len() * 48);
    for mi in &serial {
        rec.extend_from_slice(&mi.addr.to_le_bytes());
        rec.extend_from_slice(&(mi.is_write as u32).to_le_bytes());
        rec.extend_from_slice(&(mi.width as u32).to_le_bytes());
        rec.extend_from_slice(&0u32.to_le_bytes()); // pad
        rec.extend_from_slice(&mi.step.to_le_bytes());
        rec.extend_from_slice(&mi.value.to_le_bytes());
        rec.extend_from_slice(&mi.mem_values[0].to_le_bytes());
        rec.extend_from_slice(&mi.mem_values[1].to_le_bytes());
    }
    fs::write(out.join("input_records.bin"), &rec)?;
    let meta = serde_json::json!({
        "chip": "mem_align",
        "case": "fullprogram",
        "source_elf": elf_path,
        "air": "MemAlign (air_id 17)",
        "input_count": serial.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "record_layout": "repr(C) { addr:u32, is_write:u32, width:u32, pad:u32, step:u64, value:u64, mem_values:[u64;2] } (48 bytes)",
        "num_rows": num_rows,
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote mem_align/fullprogram: {} ({rows} rows x {n_cols} cols, {} inputs)\n  golden_sha256: {golden_sha256}",
        out.display(),
        serial.len(),
    );
    if debug_trace {
        write_trace_dump(&out.join("expected_mem_align_trace.npy.gz"), &air.trace, rows, n_cols)?;
        println!("  + expected_mem_align_trace.npy.gz (golden preimage, not committed)");
    }
    Ok(())
}

/// Slice of the segmented-mem oracle (#1845): emit the MemAlign byte-variant
/// (air_id 19 ReadByte / 20 WriteByte) full-program fixtures. Same count→plan→
/// expand shape as MemAlign but byte-granular and **multi-segment** — the byte
/// AIRs carry no previous-segment, so each segment is independent (no carry to
/// thread). Records are the same 48-byte `MemAlignInput` stream rw's
/// `HashZiskMemAlign{Read,Write}ByteTrace` consume. Defined via a macro because
/// read/write differ only in the concrete instance/trace types + air id.
macro_rules! emit_mem_align_byte_variant {
    ($fn_name:ident, $instance:ident, $build_collector:ident, $trace:ident, $row:ident,
     $air_ids:expr, $chip:literal, $air_label:literal) => {
        #[allow(clippy::too_many_arguments)]
        fn $fn_name(
            elf_path: &str,
            inputs_path: Option<&str>,
            out: &Path,
            debug_trace: bool,
            std: Arc<Std<F>>,
            sctx: Arc<SetupCtx<F>>,
            pctx: Arc<ProofCtx<F>>,
        ) -> Result<()> {
            let elf_bytes = fs::read(elf_path)?;
            let rom: ZiskRom = Riscv2zisk::new(&elf_bytes)
                .run()
                .map_err(|e| anyhow::anyhow!("riscv2zisk transpile failed: {e}"))?;
            let input_data: Vec<u8> = match inputs_path {
                Some(p) => fs::read(p)?,
                None => Vec::new(),
            };
            let emu_options = EmuOptions {
                chunk_size: Some(1 << 18),
                max_steps: 0xF_FFFF_FFFF,
                ..EmuOptions::default()
            };
            let min_traces =
                ZiskEmulator::compute_minimal_traces(&rom, &input_data, &emu_options, 1)?;

            let mut metrics: Vec<(ChunkId, Box<dyn BusDeviceMetrics>)> = Vec::new();
            for (i, emu_trace) in min_traces.iter().enumerate() {
                let mut bus = MemCountBus { counters: MemCounters::new(), mem_writes_seen: 0 };
                // Chunk-0 counter carries the memory-init sections, as in the
                // executor (execution/rust.rs `is_first()`); the mem_align
                // counters inside are untouched by it, but the shape matches
                // production.
                if i == 0 {
                    bus.counters.init_with_mem_sections(&rom as &dyn zisk_core::MemDataSection);
                }
                ZiskEmulator::process_emu_trace::<F, (), MemCountBus>(
                    &rom, emu_trace, &mut bus, true,
                );
                bus.counters.close();
                metrics.push((ChunkId(i), Box::new(bus.counters)));
            }
            let mut plans: Vec<_> = MemPlanner::new()
                .plan(metrics)
                .into_iter()
                .filter(|p| p.air_id == $air_ids[0])
                .collect();
            if plans.is_empty() {
                bail!("planner produced no {} plan", $air_label);
            }
            plans.sort_by_key(|p| p.segment_id.map(|s| s.0).unwrap_or(0));
            let total_segments = plans.len();

            for plan in plans {
                let seg = plan.segment_id.map(|s| s.0).unwrap_or(0);
                let chunk_ids = plan_chunk_ids(&plan.check_point, $air_label)?;
                let instance =
                    $instance::new(MemAlignByteSM::new(std.clone()), InstanceCtx::new(0, plan));
                let mut collectors: Vec<(usize, Box<dyn BusDevice<u64>>)> = Vec::new();
                let mut serial: Vec<MemAlignInput> = Vec::new();
                for cid in &chunk_ids {
                    let collector = instance.$build_collector(*cid);
                    let mut bus = MemAlignFeedBus { collector };
                    ZiskEmulator::process_emu_trace::<F, (), MemAlignFeedBus>(
                        &rom,
                        &min_traces[cid.0],
                        &mut bus,
                        true,
                    );
                    let collector = bus.collector;
                    for mi in &collector.inputs {
                        serial.push(MemAlignInput {
                            addr: mi.addr,
                            is_write: mi.is_write,
                            width: mi.width,
                            step: mi.step,
                            value: mi.value,
                            mem_values: mi.mem_values,
                        });
                    }
                    collectors.push((cid.0, Box::new(collector) as Box<dyn BusDevice<u64>>));
                }

                let num_rows = $trace::<$row<F>>::NUM_ROWS;
                let buffer = vec![F::default(); num_rows * $trace::<$row<F>>::ROW_SIZE];
                let air = instance
                    .compute_witness(&pctx, &sctx, collectors, buffer, false)?
                    .ok_or_else(|| {
                        anyhow::anyhow!("{} instance produced no AirInstance", $air_label)
                    })?;
                let n_cols = air.n_cols_trace;
                let rows = air.trace.len() / n_cols;
                let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();
                let mut hasher = Sha256::new();
                for v in &data {
                    hasher.update(v.to_le_bytes());
                }
                let golden_sha256 =
                    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

                let seg_out = out.join(format!("seg{seg:02}"));
                fs::create_dir_all(&seg_out)?;
                let mut rec = Vec::with_capacity(serial.len() * 48);
                for mi in &serial {
                    rec.extend_from_slice(&mi.addr.to_le_bytes());
                    rec.extend_from_slice(&(mi.is_write as u32).to_le_bytes());
                    rec.extend_from_slice(&(mi.width as u32).to_le_bytes());
                    rec.extend_from_slice(&0u32.to_le_bytes()); // pad
                    rec.extend_from_slice(&mi.step.to_le_bytes());
                    rec.extend_from_slice(&mi.value.to_le_bytes());
                    rec.extend_from_slice(&mi.mem_values[0].to_le_bytes());
                    rec.extend_from_slice(&mi.mem_values[1].to_le_bytes());
                }
                fs::write(seg_out.join("input_records.bin"), &rec)?;
                let meta = serde_json::json!({
                    "chip": $chip,
                    "case": "fullprogram",
                    "source_elf": elf_path,
                    "air": $air_label,
                    "segment_id": seg,
                    "total_segments": total_segments,
                    "input_count": serial.len(),
                    "trace_rows": rows,
                    "trace_cols": n_cols,
                    "field": "goldilocks_canonical_u64",
                    "golden_sha256": golden_sha256,
                    "record_layout": "repr(C) { addr:u32, is_write:u32, width:u32, pad:u32, step:u64, value:u64, mem_values:[u64;2] } (48 bytes)",
                    "num_rows": num_rows,
                });
                fs::write(
                    seg_out.join("fixture_metadata.json"),
                    serde_json::to_string_pretty(&meta)?,
                )?;
                println!(
                    "wrote {}/fullprogram seg {seg}/{total_segments}: {} ({rows} rows x {n_cols} cols, {} inputs)\n  golden_sha256: {golden_sha256}",
                    $chip,
                    seg_out.display(),
                    serial.len(),
                );
                if debug_trace {
                    let name = format!("expected_{}_trace.npy.gz", $chip);
                    write_npy_gz(&seg_out.join(&name), &data, rows, n_cols)?;
                    println!("  + {name} (golden preimage, not committed)");
                }
            }
            Ok(())
        }
    };
}

emit_mem_align_byte_variant!(
    emit_mem_align_byte,
    MemAlignByteInstance,
    build_mem_align_byte_collector,
    MemAlignByteTrace,
    MemAlignByteTraceRow,
    MEM_ALIGN_BYTE_AIR_IDS,
    "mem_align_byte",
    "MemAlignByte (air_id 6)"
);

emit_mem_align_byte_variant!(
    emit_mem_align_read_byte,
    MemAlignReadByteInstance,
    build_mem_align_read_byte_collector,
    MemAlignReadByteTrace,
    MemAlignReadByteTraceRow,
    MEM_ALIGN_READ_BYTE_AIR_IDS,
    "mem_align_read_byte",
    "MemAlignReadByte (air_id 7)"
);

emit_mem_align_byte_variant!(
    emit_mem_align_write_byte,
    MemAlignWriteByteInstance,
    build_mem_align_write_byte_collector,
    MemAlignWriteByteTrace,
    MemAlignWriteByteTraceRow,
    MEM_ALIGN_WRITE_BYTE_AIR_IDS,
    "mem_align_write_byte",
    "MemAlignWriteByte (air_id 8)"
);

/// Lowercase hex, no separator.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure_parent(path)?;
    fs::write(path, bytes)?;
    Ok(())
}

/// `git rev-parse HEAD` of the producer worktree. Best-effort: "unknown" if git
/// is unavailable or the worktree is detached without a resolvable HEAD. Pins
/// the producer revision into the fixture metadata.
fn worktree_head_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Dumps the ZROM + ZPIN blobs the riscv-witness ZisK perf-bench (#1917)
/// consumes. Both are pure functions of the transpiled `ZiskRom`: ZROM is the
/// serialized instruction stream, ZPIN bundles the raw guest input with the
/// ELF's read-only sections (`ZiskRom::ro_data`). No emulation is needed —
/// the perf-bench regenerates checkpoints itself from these two blobs.
fn emit_rom_input_dump(
    elf: &Path,
    input_path: Option<&Path>,
    rom_out: &Path,
    program_input_out: &Path,
    metadata_out: &Path,
    zisk_commit: Option<&str>,
) -> Result<()> {
    let elf_bytes = fs::read(elf)?;
    let input_bytes: Vec<u8> = match input_path {
        Some(p) => fs::read(p)?,
        None => Vec::new(),
    };
    let input_sha = hex(&Sha256::digest(&input_bytes));

    let rom = Riscv2zisk::new(&elf_bytes)
        .run()
        .map_err(|e| anyhow::anyhow!("transpile failed: {e}"))?;

    let mut rom_buf = Vec::new();
    zisk_wire::write_zrom(&mut rom_buf, &rom)?;
    write_bytes(rom_out, &rom_buf)?;
    let rom_sha = hex(&Sha256::digest(&rom_buf));

    // RO + RW sections come from upstream `ZiskRom::{ro_data_64, rw_data_64}`
    // (Vec<DataSection64{addr, data: Vec<u64>}>): elf2rom packs each section's
    // bytes into little-endian u64 words, so reverse that to recover the wire
    // bytes and pass `(addr, &bytes)` pairs. RW is the writable `.data` initial
    // image native loads via `Mem::init_write_section_data`; omitting it leaves
    // the consumer's RAM `.data` all-zero (rw #2067 divergence).
    let section_bytes = |sections: &[zisk_core::DataSection64]| -> Vec<(u64, Vec<u8>)> {
        sections
            .iter()
            .map(|s| (s.addr, s.data.iter().flat_map(|w| w.to_le_bytes()).collect()))
            .collect()
    };
    let ro_section_bytes = section_bytes(&rom.ro_data_64);
    let rw_section_bytes = section_bytes(&rom.rw_data_64);
    let ro_section_owned: Vec<(u64, &[u8])> =
        ro_section_bytes.iter().map(|(addr, bytes)| (*addr, bytes.as_slice())).collect();
    let rw_section_owned: Vec<(u64, &[u8])> =
        rw_section_bytes.iter().map(|(addr, bytes)| (*addr, bytes.as_slice())).collect();
    let pin = ProgramInput {
        input: &input_bytes,
        ro_sections: &ro_section_owned,
        rw_sections: &rw_section_owned,
    };
    let mut pin_buf = Vec::new();
    zisk_wire::write_zpin(&mut pin_buf, &pin)?;
    write_bytes(program_input_out, &pin_buf)?;
    let pin_sha = hex(&Sha256::digest(&pin_buf));

    // Prefer the caller-supplied commit (the genrule passes it — its cwd is the
    // consumer execroot, not this fork). Fall back to the worktree HEAD only for
    // a standalone run from the fork, where that resolves correctly.
    let commit = zisk_commit.map(str::to_string).unwrap_or_else(worktree_head_sha);
    let meta = serde_json::json!({
        "zisk_commit": commit,
        "elf": elf.display().to_string(),
        "input_sha256": input_sha,
        "rom_sha256": rom_sha,
        "program_input_sha256": pin_sha,
    });
    ensure_parent(metadata_out)?;
    fs::write(metadata_out, serde_json::to_string_pretty(&meta)?)?;

    println!(
        "wrote rom/program_input bundle:\n  rom            = {} ({} B, {} insts)\n  program_input  = {} ({} B)\n  metadata       = {}",
        rom_out.display(),
        rom_buf.len(),
        rom.sorted_pc_list.len(),
        program_input_out.display(),
        pin_buf.len(),
        metadata_out.display(),
    );
    Ok(())
}

/// Per-chunk execute-seed reference oracle (#2067) — the live subphase-1
/// analogue of the subphase-3 trace-parity harness (#1508). Runs the emulator
/// over the guest and writes, for each checkpoint chunk, the start pc / clk /
/// 32-register seed in the zkVM-agnostic schema riscv-witness'
/// `riscv_witness/testing/checkpoint_seed_parity.h` parses via
/// `LoadCheckpointSeedReference`: `{ "checkpoints": [ { step_count, start_pc,
/// start_clk, start_registers[32], mem_read_values } ] }`, every value a u64.
///
/// One `checkpoints[]` entry per minimal trace: the chunk boundary is
/// `chunk_size` steps, which must match the consumer's
/// `CheckpointConfig::checkpoint_size` so the seeds line up with the rw
/// checkpoint stream (the existing `zisk_checkpoint_aot_parity_test` already
/// byte-matches that stream against the fork's per-chunk checkpoint dump, so the
/// chunk count is the checkpoint count). `mem_read_values` is informational on
/// the rw side — `ExpectCheckpointSeedMatches` does not assert it — so it is
/// emitted empty.
fn emit_checkpoint_reference(
    elf_path: &str,
    inputs_path: Option<&str>,
    chunk_size: u64,
    out: &Path,
) -> Result<()> {
    let (_rom, min_traces) =
        load_guest_min_traces_with_chunk_size(elf_path, inputs_path, chunk_size)?;

    let checkpoints: Vec<serde_json::Value> = min_traces
        .iter()
        .map(|t| {
            serde_json::json!({
                "step_count": t.steps,
                "start_pc": t.start_state.pc,
                "start_clk": t.start_state.step,
                "start_registers": t.start_state.regs.to_vec(),
                "mem_read_values": Vec::<u64>::new(),
            })
        })
        .collect();
    let total_steps: u64 = min_traces.iter().map(|t| t.steps).sum();
    let doc = serde_json::json!({ "checkpoints": checkpoints });

    ensure_parent(out)?;
    fs::write(out, serde_json::to_string_pretty(&doc)?)?;
    println!(
        "wrote checkpoint seed reference: {} ({} checkpoint(s), chunk_size={chunk_size}, {total_steps} total steps)",
        out.display(),
        min_traces.len(),
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // `--emu-trace-dump` is proofman-free (transpile only) — dispatch it before
    // `build_std` so it needs no proving key / setup context.
    if args.emu_trace_dump {
        let elf = args
            .elf
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--emu-trace-dump requires --elf"))?;
        let rom_out = args
            .rom_out
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--emu-trace-dump requires --rom-out"))?;
        let program_input_out = args
            .program_input_out
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--emu-trace-dump requires --program-input-out"))?;
        let metadata_out = args
            .metadata_out
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--emu-trace-dump requires --metadata-out"))?;
        return emit_rom_input_dump(
            Path::new(elf),
            args.inputs.as_deref().map(Path::new),
            Path::new(rom_out),
            Path::new(program_input_out),
            Path::new(metadata_out),
            args.zisk_commit.as_deref(),
        );
    }

    // `--checkpoint-reference` is proofman-free (transpile + emulate only) — like
    // `--emu-trace-dump`, dispatch it before `build_std` so it needs no proving key.
    if let Some(out) = args.checkpoint_reference.as_deref() {
        let elf = args
            .elf
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--checkpoint-reference requires --elf"))?;
        let chunk_size = args.chunk_size.unwrap_or(1 << 18);
        if chunk_size == 0 {
            bail!("--chunk-size must be > 0");
        }
        return emit_checkpoint_reference(
            elf,
            args.inputs.as_deref(),
            chunk_size,
            Path::new(out),
        );
    }

    if args.hash_family.is_some() && args.stage1_root.is_none() {
        bail!("--hash-family is only meaningful with --stage1-root");
    }
    let (std, sctx, pctx) = build_std(args.hash_family.as_deref())?;

    if let Some(fixture_dir) = args.stage1_root.as_deref() {
        return emit_stage1_root(Path::new(fixture_dir), &sctx, &pctx);
    }

    if let Some(elf) = args.elf.as_deref() {
        if args.mem_spike {
            return mem_spike(elf, args.inputs.as_deref());
        }
        if args.mem {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_mem(elf, args.inputs.as_deref(), &Path::new(out).join("mem/fullprogram"), args.debug_trace, std, sctx, pctx);
        }
        if args.mem_align {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_mem_align(elf, args.inputs.as_deref(), &Path::new(out).join("mem_align/fullprogram"), args.debug_trace, std, sctx, pctx);
        }
        if args.mem_align_byte {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_mem_align_byte(elf, args.inputs.as_deref(), &Path::new(out).join("mem_align_byte/fullprogram"), args.debug_trace, std, sctx, pctx);
        }
        if args.mem_align_read_byte {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_mem_align_read_byte(elf, args.inputs.as_deref(), &Path::new(out).join("mem_align_read_byte/fullprogram"), args.debug_trace, std, sctx, pctx);
        }
        if args.mem_align_write_byte {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_mem_align_write_byte(elf, args.inputs.as_deref(), &Path::new(out).join("mem_align_write_byte/fullprogram"), args.debug_trace, std, sctx, pctx);
        }
        if args.binary_multi {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_binary_multi(elf, args.inputs.as_deref(), Path::new(out), args.debug_trace, std, sctx, pctx);
        }
        if args.keccak_multi {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_keccak_multi(elf, args.inputs.as_deref(), Path::new(out), args.debug_trace, std, sctx);
        }
        if args.main_multi {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_main_multi(elf, args.inputs.as_deref(), Path::new(out), args.debug_trace, std, sctx);
        }
        if args.rom_data {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_rom_data(elf, args.inputs.as_deref(), &Path::new(out).join("rom_data/fullprogram"), args.debug_trace, std, sctx, pctx);
        }
        if args.input_data {
            let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
            return emit_input_data(elf, args.inputs.as_deref(), &Path::new(out).join("input_data/fullprogram"), args.debug_trace, std, sctx, pctx);
        }
        // `--out` is the testdata base (e.g. testdata/zisk/v1); the oracle writes
        // <base>/binary/fullprogram and <base>/binary_extension/fullprogram.
        let out = args.out.as_deref().unwrap_or("/tmp/rw-fullprogram");
        return emit_fullprogram(elf, args.inputs.as_deref(), Path::new(out), args.debug_trace, std, sctx);
    }

    if args.selftest {
        let _sm = BinaryBasicSM::<F>::new(std);
        println!("setup ok: ProofCtx + SetupCtx + Std built; BinaryBasicSM constructed");
        return Ok(());
    }

    match (args.chip.as_deref(), args.case.as_deref(), args.out.as_deref()) {
        (Some("binary"), Some(case), Some(out)) => {
            emit_binary(case, Path::new(out), args.debug_trace, std)
        }
        (Some("binary_extension"), Some(case), Some(out)) => {
            emit_binary_extension(case, Path::new(out), args.debug_trace, std)
        }
        (Some("arith_eq"), Some(case), Some(out)) => {
            emit_arith_eq(case, Path::new(out), args.debug_trace, std, sctx)
        }
        (Some("keccak"), Some(case), Some(out)) => {
            emit_keccak(case, Path::new(out), args.debug_trace, std, sctx)
        }
        (Some("sha256"), Some(case), Some(out)) => {
            emit_sha256(case, Path::new(out), args.debug_trace, std, sctx)
        }
        _ => bail!(
            "usage: --selftest | --chip {{binary,binary_extension,arith_eq,keccak,sha256}} \
             --case <name> --out <dir>"
        ),
    }
}

fn emit_binary_extension(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
) -> Result<()> {
    let records = case_inputs_extension(case)?;
    let inputs: Vec<Vec<BinaryInput>> =
        vec![records.iter().map(|r| BinaryInput::new(r.op, r.a, r.b)).collect()];

    let sm = BinaryExtensionSM::<F>::new(std);
    let num_rows = BinaryExtensionTrace::<BinaryExtensionTraceRow<F>>::NUM_ROWS;
    let row_size = BinaryExtensionTrace::<BinaryExtensionTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<BinaryExtensionTraceRow<F>>(&inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    // Golden hash — same canonical row-major u64 LE format as emit_binary.
    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;

    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.push(r.op);
        rec_bytes.extend_from_slice(&r._pad);
        rec_bytes.extend_from_slice(&r.a.to_le_bytes());
        rec_bytes.extend_from_slice(&r.b.to_le_bytes());
        rec_bytes.extend_from_slice(&r.result.to_le_bytes());
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "790f9e28a (fractalyze/zisk ref)",
        "chip": "binary_extension",
        "case": case,
        "air": "BinaryExtension (air_id 24)",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { op:u8, pad[7], a:u64, b:u64, result:u64 }",
    });
    fs::write(out.join("fixture_metadata.json"), serde_json::to_string_pretty(&meta)?)?;

    if debug_trace {
        write_npy_gz(
            &out.join("expected_binary_extension_trace.npy.gz"),
            &data,
            rows,
            n_cols,
        )?;
    }

    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace {
            "\n  + expected_binary_extension_trace.npy.gz (debug, not committed)"
        } else {
            ""
        }
    );
    Ok(())
}

/// One ArithEq `arith256` operation, mirrored byte-for-byte by the rw record.
/// repr(C), little-endian — 128 bytes, no padding (u64 step first, six u32
/// addresses fill 8..32, then three [u64;4] operands at 32/64/96). Mirrors the
/// fields of native `Arith256Input` that drive trace generation; `addr`/`*_addr`
/// only populate the multiplexed `step_addr` column, the a/b/c operands drive
/// the `a*b + c = dh*2^256 + dl` equation.
#[repr(C)]
struct ArithEqArith256Record {
    step: u64,
    addr: u32,
    a_addr: u32,
    b_addr: u32,
    c_addr: u32,
    dl_addr: u32,
    dh_addr: u32,
    a: [u64; 4],
    b: [u64; 4],
    c: [u64; 4],
}

/// ArithEq fixture inputs (op 0 = `arith256`: `a*b + c = dh:dl`, 16 rows/op).
/// The `arith256_single` case walks the carry/output corners of the 256-bit
/// multiply-accumulate, each with distinct addresses/step so the multiplexed
/// `step_addr` column is exercised too:
///   - trivial `2*3+1 = 7` (no carry, dh=0),
///   - `(2^64-1)*2` (carry across the limb-0 → limb-1 boundary),
///   - `(2^256-1)*1` (fills every `x3`/dl chunk, dh=0),
///   - `(2^256-1)*2 + 1` (high output `y3`/dh becomes 1),
///   - `(2^256-1)^2 + (2^256-1)` (max product: full carry chains, dh near 2^256).
fn case_inputs_arith_eq_arith256(case: &str) -> Result<Vec<ArithEqArith256Record>> {
    let max = u64::MAX;
    let all = [max, max, max, max];
    let rec = |step, addr, a_addr, b_addr, c_addr, dl_addr, dh_addr, a, b, c| {
        ArithEqArith256Record { step, addr, a_addr, b_addr, c_addr, dl_addr, dh_addr, a, b, c }
    };
    let v = match case {
        "arith256_single" => vec![
            rec(100, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000,
                [2, 0, 0, 0], [3, 0, 0, 0], [1, 0, 0, 0]),
            rec(200, 0x1010, 0x2010, 0x3010, 0x4010, 0x5010, 0x6010,
                [max, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]),
            rec(300, 0x1020, 0x2020, 0x3020, 0x4020, 0x5020, 0x6020,
                all, [1, 0, 0, 0], [0, 0, 0, 0]),
            rec(400, 0x1030, 0x2030, 0x3030, 0x4030, 0x5030, 0x6030,
                all, [2, 0, 0, 0], [1, 0, 0, 0]),
            rec(500, 0x1040, 0x2040, 0x3040, 0x4040, 0x5040, 0x6040, all, all, all),
        ],
        other => bail!("unknown arith256 case {other:?}"),
    };
    Ok(v)
}

/// One ArithEq `arith256_mod` operation (op 1: `a*b + c mod m = d`, 16 rows/op),
/// mirrored byte-for-byte by the rw record. repr(C), little-endian, 160 bytes:
/// u64 step first, six u32 addresses fill 8..32, then four [u64;4] operands at
/// 32/64/96/128 (a, b, c, module). Mirrors the fields of native
/// `Arith256ModInput` that drive trace generation; `addr`/`*_addr` only populate
/// the multiplexed `step_addr` column, the a/b/c/module operands drive the
/// modular `a*b + c ≡ d (mod module)` equation (with quotient `q = q1*2^256 + q0`).
#[repr(C)]
struct ArithEqArith256ModRecord {
    step: u64,
    addr: u32,
    a_addr: u32,
    b_addr: u32,
    c_addr: u32,
    module_addr: u32,
    d_addr: u32,
    a: [u64; 4],
    b: [u64; 4],
    c: [u64; 4],
    module: [u64; 4],
}

/// ArithEq fixture inputs for op 1 = `arith256_mod`: `d = (a*b + c) mod m`.
/// Walks the corners that exercise both q0 (low 256 bits of quotient) and q1
/// (the carry above 2^256, needed when `a*b + c >= 2^256 * m`):
///   - `2*3 + 1 mod 5 = 2` (trivial, q0 small, q1=0, x3<y2 on row 0),
///   - `(2^64-1)*2 + 1 mod 2^64 = 2^64-1` (single-limb carry, q0=1, q1=0),
///   - `100*200 + 50 mod 1024 = 594` (medium values, q0=19, q1=0),
///   - `(2^256-1)*1 + 0 mod 2 = 1` (full-width q0 = 2^255-1, q1=0),
///   - `(2^256-1)^2 + 0 mod 1 = 0` (q1=2^256-2 needed: exercises high-half quotient).
fn case_inputs_arith_eq_arith256_mod(case: &str) -> Result<Vec<ArithEqArith256ModRecord>> {
    let max = u64::MAX;
    let all = [max, max, max, max];
    let rec = |step, addr, a_addr, b_addr, c_addr, module_addr, d_addr, a, b, c, module| {
        ArithEqArith256ModRecord {
            step,
            addr,
            a_addr,
            b_addr,
            c_addr,
            module_addr,
            d_addr,
            a,
            b,
            c,
            module,
        }
    };
    let v = match case {
        "arith256_mod_single" => vec![
            // 2*3+1 = 7; 7 mod 5 = 2. x3=2, q0=1, q1=0.
            rec(100, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000,
                [2, 0, 0, 0], [3, 0, 0, 0], [1, 0, 0, 0], [5, 0, 0, 0]),
            // (2^64-1)*2 + 1 = 2^65 - 1; mod 2^64 = 2^64 - 1. x3=2^64-1, q0=1, q1=0.
            rec(200, 0x1010, 0x2010, 0x3010, 0x4010, 0x5010, 0x6010,
                [max, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]),
            // 100*200+50 = 20050; mod 1024 = 594. q0=19, q1=0.
            rec(300, 0x1020, 0x2020, 0x3020, 0x4020, 0x5020, 0x6020,
                [100, 0, 0, 0], [200, 0, 0, 0], [50, 0, 0, 0], [1024, 0, 0, 0]),
            // (2^256-1)*1 + 0 mod 2 = 1. q0 = 2^255-1, q1 = 0.
            rec(400, 0x1030, 0x2030, 0x3030, 0x4030, 0x5030, 0x6030,
                all, [1, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]),
            // (2^256-1)^2 + 0 mod 1 = 0. q = (2^256-1)^2 = 2^512 - 2^257 + 1 ⇒
            // q0 = 1, q1 = 2^256 - 2 (exercises the q1*y2 cross-term).
            rec(500, 0x1040, 0x2040, 0x3040, 0x4040, 0x5040, 0x6040,
                all, all, [0, 0, 0, 0], [1, 0, 0, 0]),
        ],
        other => bail!("unknown arith256_mod case {other:?}"),
    };
    Ok(v)
}

/// One ArithEq `secp256k1_add` operation (op 2: p3 = p1 + p2 on secp256k1).
/// Mirrored byte-for-byte by the rw `ZiskArithEqSecp256k1AddOpRecord`. repr(C),
/// little-endian — 152 bytes. The explicit `_pad` u32 keeps the u64 point
/// arrays 8-aligned so stride == sizeof on both sides; the parser skips it.
/// Points are `[u64;8]` = (x ‖ y), 4 LE limbs per coord; mirrors native
/// `Secp256k1AddInput`.
#[repr(C)]
struct ArithEqSecp256k1AddRecord {
    step: u64,
    addr: u32,
    p1_addr: u32,
    p2_addr: u32,
    _pad: u32,
    p1: [u64; 8],
    p2: [u64; 8],
}

/// One ArithEq `secp256k1_dbl` operation (op 3: p3 = 2*p1). repr(C),
/// little-endian, 80 bytes. Mirrors native `Secp256k1DblInput` (no second
/// point, no point addresses — every coord address is `addr` or `addr+32`).
#[repr(C)]
struct ArithEqSecp256k1DblRecord {
    step: u64,
    addr: u32,
    _pad: u32,
    p1: [u64; 8],
}

// secp256k1 generator G and 2G in 4 little-endian u64 limbs per coordinate.
//
// G  = (Gx,  Gy):
//   Gx  = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
//   Gy  = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
// 2G = (2Gx, 2Gy):
//   2Gx = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
//   2Gy = 0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A
const SECP256K1_G_X: [u64; 4] = [
    0x59F2815B16F81798,
    0x029BFCDB2DCE28D9,
    0x55A06295CE870B07,
    0x79BE667EF9DCBBAC,
];
const SECP256K1_G_Y: [u64; 4] = [
    0x9C47D08FFB10D4B8,
    0xFD17B448A6855419,
    0x5DA4FBFC0E1108A8,
    0x483ADA7726A3C465,
];
const SECP256K1_2G_X: [u64; 4] = [
    0xABAC09B95C709EE5,
    0x5C778E4B8CEF3CA7,
    0x3045406E95C07CD8,
    0xC6047F9441ED7D6D,
];
const SECP256K1_2G_Y: [u64; 4] = [
    0x236431A950CFE52A,
    0xF7F632653266D0E1,
    0xA3C58419466CEAEE,
    0x1AE168FEA63DC339,
];

fn point(x: [u64; 4], y: [u64; 4]) -> [u64; 8] {
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}

/// ArithEq fixture inputs for op 2 = `secp256k1_add`. `add_single` adds the
/// secp256k1 generator G to 2G — well-known on-curve points that exercise the
/// add path: full-width field operands, x1 != x2 (so x_are_different fires on
/// row 0), the modular inverse for `s = (y2-y1)/(x2-x1)`, and the
/// offset-adjusted quotients q0/q1/q2.
fn case_inputs_arith_eq_secp256k1_add(case: &str) -> Result<Vec<ArithEqSecp256k1AddRecord>> {
    let v = match case {
        "secp256k1_add_single" => vec![ArithEqSecp256k1AddRecord {
            step: 100,
            addr: 0x1000,
            p1_addr: 0x2000,
            p2_addr: 0x3000,
            _pad: 0,
            p1: point(SECP256K1_G_X, SECP256K1_G_Y),
            p2: point(SECP256K1_2G_X, SECP256K1_2G_Y),
        }],
        other => bail!("unknown secp256k1_add case {other:?}"),
    };
    Ok(v)
}

/// ArithEq fixture inputs for op 3 = `secp256k1_dbl`. `dbl_single` doubles the
/// generator G — exercises the dbl-only slope `s = 3*x1^2/(2*y1)` and offset
/// 2^258 on q0 (vs add's 2^257).
fn case_inputs_arith_eq_secp256k1_dbl(case: &str) -> Result<Vec<ArithEqSecp256k1DblRecord>> {
    let v = match case {
        "secp256k1_dbl_single" => vec![ArithEqSecp256k1DblRecord {
            step: 100,
            addr: 0x1000,
            _pad: 0,
            p1: point(SECP256K1_G_X, SECP256K1_G_Y),
        }],
        other => bail!("unknown secp256k1_dbl case {other:?}"),
    };
    Ok(v)
}

/// One ArithEq `secp256r1_add` operation (op 9: p3 = p1 + p2 on secp256r1 /
/// P-256). Same repr(C) layout as `ArithEqSecp256k1AddRecord` — 152 bytes,
/// mirrors native `Secp256r1AddInput`.
#[repr(C)]
struct ArithEqSecp256r1AddRecord {
    step: u64,
    addr: u32,
    p1_addr: u32,
    p2_addr: u32,
    _pad: u32,
    p1: [u64; 8],
    p2: [u64; 8],
}

/// One ArithEq `secp256r1_dbl` operation (op 10: p3 = 2*p1). repr(C),
/// little-endian, 80 bytes. Mirrors native `Secp256r1DblInput`.
#[repr(C)]
struct ArithEqSecp256r1DblRecord {
    step: u64,
    addr: u32,
    _pad: u32,
    p1: [u64; 8],
}

// secp256r1 (NIST P-256) generator G and 2G, 4 little-endian u64 limbs per
// coordinate. Computed + on-curve-verified (y^2 = x^3 - 3x + b) offline.
//   Gx  = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
//   Gy  = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
//   2Gx = 0x7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978
//   2Gy = 0x07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1
const SECP256R1_G_X: [u64; 4] =
    [0xf4a13945d898c296, 0x77037d812deb33a0, 0xf8bce6e563a440f2, 0x6b17d1f2e12c4247];
const SECP256R1_G_Y: [u64; 4] =
    [0xcbb6406837bf51f5, 0x2bce33576b315ece, 0x8ee7eb4a7c0f9e16, 0x4fe342e2fe1a7f9b];
const SECP256R1_2G_X: [u64; 4] =
    [0xa60b48fc47669978, 0xc08969e277f21b35, 0x8a52380304b51ac3, 0x7cf27b188d034f7e];
const SECP256R1_2G_Y: [u64; 4] =
    [0x9e04b79d227873d1, 0xba7dade63ce98229, 0x293d9ac69f7430db, 0x07775510db8ed040];

/// ArithEq fixture inputs for op 9 = `secp256r1_add`: G + 2G on P-256. Twin of
/// the secp256k1 add case; exercises the `a = -3` curve constant via the shared
/// add path (x1 != x2, slope inverse, offset-adjusted q0/q1/q2).
fn case_inputs_arith_eq_secp256r1_add(case: &str) -> Result<Vec<ArithEqSecp256r1AddRecord>> {
    let v = match case {
        "secp256r1_add_single" => vec![ArithEqSecp256r1AddRecord {
            step: 100,
            addr: 0x1000,
            p1_addr: 0x2000,
            p2_addr: 0x3000,
            _pad: 0,
            p1: point(SECP256R1_G_X, SECP256R1_G_Y),
            p2: point(SECP256R1_2G_X, SECP256R1_2G_Y),
        }],
        other => bail!("unknown secp256r1_add case {other:?}"),
    };
    Ok(v)
}

/// ArithEq fixture inputs for op 10 = `secp256r1_dbl`: 2*G on P-256. Exercises
/// the dbl-only slope `s = (3*x1^2 - 3)/(2*y1)` (the `-3 = a` term, absent on
/// secp256k1) and offset 2^258 on q0.
fn case_inputs_arith_eq_secp256r1_dbl(case: &str) -> Result<Vec<ArithEqSecp256r1DblRecord>> {
    let v = match case {
        "secp256r1_dbl_single" => vec![ArithEqSecp256r1DblRecord {
            step: 100,
            addr: 0x1000,
            _pad: 0,
            p1: point(SECP256R1_G_X, SECP256R1_G_Y),
        }],
        other => bail!("unknown secp256r1_dbl case {other:?}"),
    };
    Ok(v)
}

/// One ArithEq `bn254_curve_add` operation (op 4: p3 = p1 + p2 on the bn254
/// G1 curve). Same repr(C) layout as `ArithEqSecp256k1AddRecord` — 152 bytes,
/// mirrors native `Bn254CurveAddInput`.
#[repr(C)]
struct ArithEqBn254CurveAddRecord {
    step: u64,
    addr: u32,
    p1_addr: u32,
    p2_addr: u32,
    _pad: u32,
    p1: [u64; 8],
    p2: [u64; 8],
}

/// One ArithEq `bn254_curve_dbl` operation (op 5: p3 = 2*p1). repr(C),
/// little-endian, 80 bytes. Mirrors native `Bn254CurveDblInput`.
#[repr(C)]
struct ArithEqBn254CurveDblRecord {
    step: u64,
    addr: u32,
    _pad: u32,
    p1: [u64; 8],
}

// bn254 G1 generator G = (1, 2) and 2G, 4 little-endian u64 limbs per
// coordinate (base field y^2 = x^3 + 3). 2G computed + on-curve-verified
// offline; matches the well-known alt_bn128 value.
//   2Gx = 0x030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3
//   2Gy = 0x15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4
const BN254_G_X: [u64; 4] = [0x0000000000000001, 0, 0, 0];
const BN254_G_Y: [u64; 4] = [0x0000000000000002, 0, 0, 0];
const BN254_2G_X: [u64; 4] =
    [0xd3c208c16d87cfd3, 0xd97816a916871ca8, 0x9b85045b68181585, 0x030644e72e131a02];
const BN254_2G_Y: [u64; 4] =
    [0xff3ebf7a5a18a2c4, 0x68a6a449e3538fc7, 0xe7845f96b2ae9c0a, 0x15ed738c0e0a7c92];

/// ArithEq fixture inputs for op 4 = `bn254_curve_add`: G + 2G on bn254 G1.
/// Twin of the secp add cases; bn254 has `a = 0` (like secp256k1) but distinct
/// quotient offsets (add q0 2^259, q1 16, q2 2^259).
fn case_inputs_arith_eq_bn254_curve_add(case: &str) -> Result<Vec<ArithEqBn254CurveAddRecord>> {
    let v = match case {
        "bn254_curve_add_single" => vec![ArithEqBn254CurveAddRecord {
            step: 100,
            addr: 0x1000,
            p1_addr: 0x2000,
            p2_addr: 0x3000,
            _pad: 0,
            p1: point(BN254_G_X, BN254_G_Y),
            p2: point(BN254_2G_X, BN254_2G_Y),
        }],
        other => bail!("unknown bn254_curve_add case {other:?}"),
    };
    Ok(v)
}

/// ArithEq fixture inputs for op 5 = `bn254_curve_dbl`: 2*G on bn254 G1.
/// Exercises the dbl slope `s = 3*x1^2/(2*y1)` and offset 2^260 on q0.
fn case_inputs_arith_eq_bn254_curve_dbl(case: &str) -> Result<Vec<ArithEqBn254CurveDblRecord>> {
    let v = match case {
        "bn254_curve_dbl_single" => vec![ArithEqBn254CurveDblRecord {
            step: 100,
            addr: 0x1000,
            _pad: 0,
            p1: point(BN254_G_X, BN254_G_Y),
        }],
        other => bail!("unknown bn254_curve_dbl case {other:?}"),
    };
    Ok(v)
}

/// One ArithEq `bn254_complex_{add,sub,mul}` operation (ops 6/7/8: Fp2
/// arithmetic over the bn254 base field). Same repr(C) layout as
/// `ArithEqBn254CurveAddRecord` — 152 bytes; the three ops share it, mirroring
/// native `Bn254Complex{Add,Sub,Mul}Input` (operands `f = (real ‖ imag)`).
#[repr(C)]
struct ArithEqBn254ComplexRecord {
    step: u64,
    addr: u32,
    f1_addr: u32,
    f2_addr: u32,
    _pad: u32,
    f1: [u64; 8],
    f2: [u64; 8],
}

// bn254 3G coordinates, reused here purely as two full-width Fp2 operands (the
// complex ops have no curve constraint); guaranteed < p. f1 = 2G, f2 = 3G gives
// distinct full-width components so mul wraps mod p in both x3 and y3.
const BN254_3G_X: [u64; 4] =
    [0xf2d355961915abf0, 0x9315d84715b8e679, 0xf40232bcb1b6bd15, 0x0769bf9ac56bea3f];
const BN254_3G_Y: [u64; 4] =
    [0xcdf1ff3dd9fe2261, 0x319e63b40b9c5b57, 0x554fdb7c8d086475, 0x2ab799bee0489429];

/// ArithEq fixture inputs for the bn254 complex ops (op 6 add / 7 sub / 8 mul).
/// All three use the same operands f1 = 2G, f2 = 3G (as Fp2 elements `a+bi`).
fn case_inputs_arith_eq_bn254_complex(case: &str) -> Result<Vec<ArithEqBn254ComplexRecord>> {
    match case {
        "bn254_complex_add_single" | "bn254_complex_sub_single" | "bn254_complex_mul_single" => {
            Ok(vec![ArithEqBn254ComplexRecord {
                step: 100,
                addr: 0x1000,
                f1_addr: 0x2000,
                f2_addr: 0x3000,
                _pad: 0,
                f1: point(BN254_2G_X, BN254_2G_Y),
                f2: point(BN254_3G_X, BN254_3G_Y),
            }])
        }
        other => bail!("unknown bn254_complex case {other:?}"),
    }
}

fn emit_arith_eq(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    match case {
        "arith256_single" => emit_arith_eq_arith256(case, out, debug_trace, std, sctx),
        "arith256_mod_single" => emit_arith_eq_arith256_mod(case, out, debug_trace, std, sctx),
        "secp256k1_add_single" => emit_arith_eq_secp256k1_add(case, out, debug_trace, std, sctx),
        "secp256k1_dbl_single" => emit_arith_eq_secp256k1_dbl(case, out, debug_trace, std, sctx),
        "secp256r1_add_single" => emit_arith_eq_secp256r1_add(case, out, debug_trace, std, sctx),
        "secp256r1_dbl_single" => emit_arith_eq_secp256r1_dbl(case, out, debug_trace, std, sctx),
        "bn254_curve_add_single" => emit_arith_eq_bn254_curve_add(case, out, debug_trace, std, sctx),
        "bn254_curve_dbl_single" => emit_arith_eq_bn254_curve_dbl(case, out, debug_trace, std, sctx),
        "bn254_complex_add_single" | "bn254_complex_sub_single" | "bn254_complex_mul_single" => {
            emit_arith_eq_bn254_complex(case, out, debug_trace, std, sctx)
        }
        other => bail!("unknown arith_eq case {other:?}"),
    }
}

fn emit_arith_eq_arith256(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_arith256(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Arith256(Arith256Input {
                addr: r.addr,
                a_addr: r.a_addr,
                b_addr: r.b_addr,
                c_addr: r.c_addr,
                dh_addr: r.dh_addr,
                dl_addr: r.dl_addr,
                step: r.step,
                a: r.a,
                b: r.b,
                c: r.c,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    // Golden hash — same canonical row-major u64 LE format as emit_binary.
    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;

    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.a_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.b_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.c_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.dl_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.dh_addr.to_le_bytes());
        for x in r.a.iter().chain(&r.b).chain(&r.c) {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op arith256",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, a_addr:u32, b_addr:u32, c_addr:u32, dl_addr:u32, dh_addr:u32, a:[u64;4], b:[u64;4], c:[u64;4] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    // Append "\n" so the file ends in a newline — keeps rw's end-of-file-fixer
    // pre-commit hook happy when the regenerated fixture is committed.
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }

    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_arith256_mod(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_arith256_mod(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Arith256Mod(Arith256ModInput {
                addr: r.addr,
                a_addr: r.a_addr,
                b_addr: r.b_addr,
                c_addr: r.c_addr,
                module_addr: r.module_addr,
                d_addr: r.d_addr,
                step: r.step,
                a: r.a,
                b: r.b,
                c: r.c,
                module: r.module,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;

    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.a_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.b_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.c_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.module_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.d_addr.to_le_bytes());
        for x in r.a.iter().chain(&r.b).chain(&r.c).chain(&r.module) {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op arith256_mod",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, a_addr:u32, b_addr:u32, c_addr:u32, module_addr:u32, d_addr:u32, a:[u64;4], b:[u64;4], c:[u64;4], module:[u64;4] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }

    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_secp256k1_add(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_secp256k1_add(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Secp256k1Add(Secp256k1AddInput {
                addr: r.addr,
                p1_addr: r.p1_addr,
                p2_addr: r.p2_addr,
                step: r.step,
                p1: r.p1,
                p2: r.p2,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.p1_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.p2_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r._pad.to_le_bytes());
        for x in r.p1.iter().chain(&r.p2) {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op secp256k1_add",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, p1_addr:u32, p2_addr:u32, _pad:u32, p1:[u64;8], p2:[u64;8] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_secp256k1_dbl(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_secp256k1_dbl(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Secp256k1Dbl(Secp256k1DblInput {
                addr: r.addr,
                step: r.step,
                p1: r.p1,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r._pad.to_le_bytes());
        for x in r.p1.iter() {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op secp256k1_dbl",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, _pad:u32, p1:[u64;8] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_secp256r1_add(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_secp256r1_add(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Secp256r1Add(Secp256r1AddInput {
                addr: r.addr,
                p1_addr: r.p1_addr,
                p2_addr: r.p2_addr,
                step: r.step,
                p1: r.p1,
                p2: r.p2,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.p1_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.p2_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r._pad.to_le_bytes());
        for x in r.p1.iter().chain(&r.p2) {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op secp256r1_add",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, p1_addr:u32, p2_addr:u32, _pad:u32, p1:[u64;8], p2:[u64;8] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_secp256r1_dbl(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_secp256r1_dbl(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Secp256r1Dbl(Secp256r1DblInput {
                addr: r.addr,
                step: r.step,
                p1: r.p1,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r._pad.to_le_bytes());
        for x in r.p1.iter() {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op secp256r1_dbl",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, _pad:u32, p1:[u64;8] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_bn254_curve_add(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_bn254_curve_add(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Bn254CurveAdd(Bn254CurveAddInput {
                addr: r.addr,
                p1_addr: r.p1_addr,
                p2_addr: r.p2_addr,
                step: r.step,
                p1: r.p1,
                p2: r.p2,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.p1_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.p2_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r._pad.to_le_bytes());
        for x in r.p1.iter().chain(&r.p2) {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op bn254_curve_add",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, p1_addr:u32, p2_addr:u32, _pad:u32, p1:[u64;8], p2:[u64;8] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_bn254_curve_dbl(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_bn254_curve_dbl(case)?;
    let inputs: Vec<Vec<ArithEqInput>> = vec![records
        .iter()
        .map(|r| {
            ArithEqInput::Bn254CurveDbl(Bn254CurveDblInput {
                addr: r.addr,
                step: r.step,
                p1: r.p1,
            })
        })
        .collect()];

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r._pad.to_le_bytes());
        for x in r.p1.iter() {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": "ArithEq (air_id 26), op bn254_curve_dbl",
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, _pad:u32, p1:[u64;8] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

fn emit_arith_eq_bn254_complex(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_arith_eq_bn254_complex(case)?;
    // Same operands across add/sub/mul; only the ArithEqInput variant differs.
    let (inputs, air_name): (Vec<Vec<ArithEqInput>>, &str) = match case {
        "bn254_complex_add_single" => (
            vec![records
                .iter()
                .map(|r| {
                    ArithEqInput::Bn254ComplexAdd(Bn254ComplexAddInput {
                        addr: r.addr,
                        f1_addr: r.f1_addr,
                        f2_addr: r.f2_addr,
                        step: r.step,
                        f1: r.f1,
                        f2: r.f2,
                    })
                })
                .collect()],
            "bn254_complex_add",
        ),
        "bn254_complex_sub_single" => (
            vec![records
                .iter()
                .map(|r| {
                    ArithEqInput::Bn254ComplexSub(Bn254ComplexSubInput {
                        addr: r.addr,
                        f1_addr: r.f1_addr,
                        f2_addr: r.f2_addr,
                        step: r.step,
                        f1: r.f1,
                        f2: r.f2,
                    })
                })
                .collect()],
            "bn254_complex_sub",
        ),
        "bn254_complex_mul_single" => (
            vec![records
                .iter()
                .map(|r| {
                    ArithEqInput::Bn254ComplexMul(Bn254ComplexMulInput {
                        addr: r.addr,
                        f1_addr: r.f1_addr,
                        f2_addr: r.f2_addr,
                        step: r.step,
                        f1: r.f1,
                        f2: r.f2,
                    })
                })
                .collect()],
            "bn254_complex_mul",
        ),
        other => bail!("unknown bn254_complex case {other:?}"),
    };

    let sm = ArithEqSM::<F>::new(std);
    let num_rows = ArithEqTrace::<ArithEqTraceRow<F>>::NUM_ROWS;
    let row_size = ArithEqTrace::<ArithEqTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<ArithEqTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.f1_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.f2_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r._pad.to_le_bytes());
        for x in r.f1.iter().chain(&r.f2) {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-arith-eq-fixture-gen (fractalyze/zisk fork)",
        "chip": "arith_eq",
        "case": case,
        "air": format!("ArithEq (air_id 26), op {air_name}"),
        "input_count": records.len(),
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) { step:u64, addr:u32, f1_addr:u32, f2_addr:u32, _pad:u32, f1:[u64;8], f2:[u64;8] }",
        "rows_per_op": 16,
        "padding_row": "default row (all zeros)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_arith_eq_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        records.len(),
        golden_sha256,
        if debug_trace { "\n  + expected_arith_eq_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

/// Keccakf fixture inputs (one keccak-f[1600] permutation per input). The
/// native `KeccakfInput` already matches the rw wire record byte-for-byte
/// (LE `step_main:u64, addr_main:u32, state:[u64;25]` = 212 bytes), so no
/// separate record struct is needed. The `keccakf_single` case walks the
/// state corners — zero state, a single low bit, distinct counting lanes,
/// all-ones saturation — each with distinct step/addr so the step/addr
/// columns are exercised too.
fn case_inputs_keccak(case: &str) -> Result<Vec<KeccakfInput>> {
    match case {
        "keccakf_single" => {
            let mut counting = [0u64; 25];
            for (i, lane) in counting.iter_mut().enumerate() {
                *lane = i as u64;
            }
            Ok(vec![
                KeccakfInput { step_main: 100, addr_main: 0x1000, state: [0; 25] },
                KeccakfInput {
                    step_main: 200,
                    addr_main: 0x1100,
                    state: {
                        let mut s = [0u64; 25];
                        s[0] = 1;
                        s
                    },
                },
                KeccakfInput { step_main: 300, addr_main: 0x1200, state: counting },
                KeccakfInput { step_main: 400, addr_main: 0x1300, state: [u64::MAX; 25] },
            ])
        }
        other => bail!("unknown keccak case {other:?}"),
    }
}

fn emit_keccak(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_keccak(case)?;

    // Serialize the wire records before the inputs are moved into the SM.
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step_main.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr_main.to_le_bytes());
        for lane in &r.state {
            rec_bytes.extend_from_slice(&lane.to_le_bytes());
        }
    }
    let input_count = records.len();
    let inputs: Vec<Vec<KeccakfInput>> = vec![records];

    let sm = KeccakfSM::<F>::new(std);
    let num_rows = KeccakfTrace::<KeccakfTraceRow<F>>::NUM_ROWS;
    let row_size = KeccakfTrace::<KeccakfTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<KeccakfTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    // zisk_commit/zisk_branch are pinned to the values in riscv-witness's
    // committed fixture (originally generated at fork commit 856b5693) so a
    // regeneration reproduces that file byte-identically.
    let meta = serde_json::json!({
        "zisk_commit": "856b56933318a504ce8f8155938729d38b911839",
        "zisk_branch": "rw/zisk-keccak-fixture (fractalyze/zisk fork)",
        "chip": "keccak",
        "case": case,
        "air": "Keccakf (air_id 28)",
        "input_count": input_count,
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) packed { step_main:u64, addr_main:u32, state:[u64;25] } = 212 bytes LE",
        "padding_row": "default row (all zeros, in_use=0)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_keccakf_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        input_count,
        golden_sha256,
        if debug_trace { "\n  + expected_keccakf_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

/// Sha256f fixture inputs (one SHA-256 compression per input). The native
/// `Sha256fInput` already matches the rw wire record byte-for-byte (LE
/// `step_main:u64, addr_main:u32, state_addr:u32, input_addr:u32,
/// state:[u64;4], input:[u64;8]` = 116 bytes), so no separate record struct
/// is needed. Every record compresses from the standard SHA-256 IV (packed
/// as h1:h0 .. h7:h6 lane pairs); the message blocks walk the corners —
/// zero block, a single low bit, counting words, all-ones saturation — each
/// with distinct step/addr so the step/addr columns are exercised too.
fn case_inputs_sha256(case: &str) -> Result<Vec<Sha256fInput>> {
    const IV: [u64; 4] =
        [0xbb67ae856a09e667, 0xa54ff53a3c6ef372, 0x9b05688c510e527f, 0x5be0cd191f83d9ab];
    let record = |i: u64, input: [u64; 8]| {
        let addr_main = 0x1000 + 0x100 * i as u32;
        Sha256fInput {
            step_main: 100 * (i + 1),
            addr_main,
            state_addr: addr_main + 0x40,
            input_addr: addr_main + 0x80,
            state: IV,
            input,
        }
    };
    match case {
        "sha256f_single" => {
            let mut counting = [0u64; 8];
            for (i, w) in counting.iter_mut().enumerate() {
                *w = i as u64;
            }
            Ok(vec![
                record(0, [0; 8]),
                record(1, {
                    let mut w = [0u64; 8];
                    w[0] = 1;
                    w
                }),
                record(2, counting),
                record(3, [u64::MAX; 8]),
            ])
        }
        other => bail!("unknown sha256 case {other:?}"),
    }
}

fn emit_sha256(
    case: &str,
    out: &Path,
    debug_trace: bool,
    std: Arc<Std<F>>,
    sctx: Arc<SetupCtx<F>>,
) -> Result<()> {
    let records = case_inputs_sha256(case)?;

    // Serialize the wire records before the inputs are moved into the SM.
    let mut rec_bytes = Vec::new();
    for r in &records {
        rec_bytes.extend_from_slice(&r.step_main.to_le_bytes());
        rec_bytes.extend_from_slice(&r.addr_main.to_le_bytes());
        rec_bytes.extend_from_slice(&r.state_addr.to_le_bytes());
        rec_bytes.extend_from_slice(&r.input_addr.to_le_bytes());
        for x in r.state.iter().chain(&r.input) {
            rec_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    let input_count = records.len();
    let inputs: Vec<Vec<Sha256fInput>> = vec![records];

    let sm = Sha256fSM::<F>::new(std);
    let num_rows = Sha256fTrace::<Sha256fTraceRow<F>>::NUM_ROWS;
    let row_size = Sha256fTrace::<Sha256fTraceRow<F>>::ROW_SIZE;
    let buffer = vec![F::default(); num_rows * row_size];

    let air = sm.compute_witness::<Sha256fTraceRow<F>>(&sctx, &inputs, buffer)?;
    let n_cols = air.n_cols_trace;
    let rows = air.trace.len() / n_cols;
    let data: Vec<u64> = air.trace.iter().map(|f| f.as_canonical_u64()).collect();

    let mut hasher = Sha256::new();
    for v in &data {
        hasher.update(v.to_le_bytes());
    }
    let golden_sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();

    fs::create_dir_all(out)?;
    fs::write(out.join("input_records.bin"), &rec_bytes)?;

    // zisk_commit/zisk_branch pin the upstream ZisK version this fixture was
    // generated against (v1.0.0-alpha = 4b9f758) so a regeneration reproduces
    // the committed file byte-identically. v1.0.0-alpha renumbered Sha256f to
    // air_id 17 (was 29) and swapped the in_use/in_use_clk_0 trace columns.
    let meta = serde_json::json!({
        "zisk_commit": "4b9f758fabc4955cac20af837019ccc31b803a46",
        "zisk_branch": "rw/zisk-v1.0.0-alpha (fractalyze/zisk fork)",
        "chip": "sha256",
        "case": case,
        "air": "Sha256f (air_id 17)",
        "input_count": input_count,
        "trace_rows": rows,
        "trace_cols": n_cols,
        "field": "goldilocks_canonical_u64",
        "golden_sha256": golden_sha256,
        "golden_hash_input": "trace as row-major canonical u64, little-endian, all rows incl padding",
        "record_layout": "repr(C) packed { step_main:u64, addr_main:u32, state_addr:u32, input_addr:u32, state:[u64;4], input:[u64;8] } = 116 bytes LE",
        "padding_row": "default row (all zeros, in_use=0)",
    });
    fs::write(
        out.join("fixture_metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;

    if debug_trace {
        write_npy_gz(&out.join("expected_sha256f_trace.npy.gz"), &data, rows, n_cols)?;
    }
    println!(
        "wrote fixture: {} ({} rows x {} cols, {} input(s))\n  golden_sha256: {}{}",
        out.display(),
        rows,
        n_cols,
        input_count,
        golden_sha256,
        if debug_trace { "\n  + expected_sha256f_trace.npy.gz (debug, not committed)" } else { "" }
    );
    Ok(())
}

//! Where to find the CUDA toolkit's NVTX interop library, which the `nvtx`
//! feature's ranges call into (`src/nvtx.rs`). A build without the feature
//! does nothing here.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var_os("CARGO_FEATURE_NVTX").is_none() {
        return;
    }
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    // The library is not on the loader's default path and a profiling build
    // runs straight out of `target/`, so carry the directory in the binary
    // rather than asking the recipe for an LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}

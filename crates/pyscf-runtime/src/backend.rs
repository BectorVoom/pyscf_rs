//! BackendKind enum + DType axis. Per CONTEXT D-04 (enum-dispatch shape)
//! and D-08 (PYSCF_DTYPE axis).
//!
//! Naming: `BackendKind` matches cintx-runtime convention (cintx-runtime
//! `options.rs` lines 16-46) — pyscf-rs uses the cintx variant rather
//! than xcfun-gpu's `Backend` because the cintx variant matches the
//! REQUIREMENTS.md spelling (FOUND-03).

/// Compiled-in backend variant. Variants outside `Cpu` are gated by the
/// corresponding workspace feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// CPU execution profile. Always available (FOUND-03 default-on).
    Cpu,
    /// CUDA via cubecl-cuda.
    #[cfg(feature = "cuda")]
    Cuda,
    /// wgpu (Vulkan/DX12/WebGPU) via cubecl-wgpu.
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// ROCm via cubecl-hip.
    #[cfg(feature = "rocm")]
    Rocm,
    /// Metal — alias for the wgpu runtime on Apple targets (D-04 metal-as-wgpu).
    #[cfg(feature = "metal")]
    Metal,
}

impl Default for BackendKind {
    /// FOUND-03: CPU is the typed default — always, infallibly.
    fn default() -> Self {
        Self::Cpu
    }
}

impl BackendKind {
    /// Stable display name used by ALG-08 log lines.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "cuda")]  Self::Cuda  => "cuda",
            #[cfg(feature = "wgpu")]  Self::Wgpu  => "wgpu",
            #[cfg(feature = "rocm")]  Self::Rocm  => "rocm",
            #[cfg(feature = "metal")] Self::Metal => "metal",
        }
    }

    /// Parse a single PYSCF_BACKEND token (case-insensitive). Returns
    /// `None` for `"auto"` (caller resolves the auto chain) and for
    /// unrecognised tokens. Per D-07: `"hip"` is an alias for `"rocm"`.
    pub fn from_env_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Self::Cuda),
            #[cfg(feature = "wgpu")]
            "wgpu" => Some(Self::Wgpu),
            #[cfg(feature = "rocm")]
            "rocm" | "hip" => Some(Self::Rocm),
            #[cfg(feature = "metal")]
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }

    /// `true` iff the env value (case-insensitive) is `"auto"`.
    pub fn is_auto_token(s: &str) -> bool {
        s.eq_ignore_ascii_case("auto")
    }
}

/// Floating-point precision axis (D-08 PYSCF_DTYPE).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
}

impl Default for DType {
    /// D-08: default is F64 (chemistry energies need it).
    fn default() -> Self {
        Self::F64
    }
}

impl DType {
    /// Read PYSCF_DTYPE; default is F64 per D-08.
    pub fn from_env() -> Self {
        match std::env::var("PYSCF_DTYPE")
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Ok("f32") => Self::F32,
            Ok("f64") => Self::F64,
            _ => Self::F64,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

//! LFM2 on CUDA, d1's decision model: a prompt pass over gated short convolutions and GQA attention, read at labels.

pub const config = @import("config.zig");
pub const weights = @import("weights.zig");

test {
    _ = config;
    _ = weights;
}

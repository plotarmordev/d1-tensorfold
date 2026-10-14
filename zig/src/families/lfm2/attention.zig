//! LFM2's prompt attention: prefill_attention.cu's flash kernel at head dim 64, over one sequence or a packed tree.
const cuda = @import("cuda");

const sym = "_ZN20tf_prefill_attention11tree_kernelILi64ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_PKiiiiiif";
/// The instantiation tree_kernel<64, 8, 4, 8>: eight warps as four query heads by two 16-row tiles.
const head_dim = 64;
const heads_per_block = 4;
const rows_per_block = 32;
const threads = 256;
const smem: u32 = 32768; // eight 32-key slots of 64 dims

pub const Attention = struct {
    mod: cuda.Module,
    kernel: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Attention {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.prefill_attention);
        errdefer mod.unload();
        return .{ .mod = mod, .kernel = try mod.function(sym) };
    }

    pub fn unload(a: *Attention) void {
        a.mod.unload();
    }

    /// Causal GQA over one sequence: q [rows, H, D], k/v [rows, KV, D] bf16 -> out [rows, H, D] bf16.
    pub fn causal(a: *const Attention, s: cuda.Stream, q: u64, k: u64, v: u64, out: u64, rows: usize, heads: usize, kv_heads: usize, dim: usize, scale: f32) !void {
        return a.tree(s, q, k, v, out, rows, 0, 0, rows, heads, kv_heads, dim, scale);
    }

    /// As causal over packed rows: a trunk, then branches (rows trunk + cu[b]..) over it and their own earlier rows.
    pub fn tree(a: *const Attention, s: cuda.Stream, q: u64, k: u64, v: u64, out: u64, trunk: usize, cu: u64, branches: usize, rows: usize, heads: usize, kv_heads: usize, dim: usize, scale: f32) !void {
        const group = heads / kv_heads;
        if (dim != head_dim or group * kv_heads != heads or group % heads_per_block != 0) return error.UnsupportedShape;
        if (rows == 0) return;
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, v, out, cu }) |p| args.add(p);
        for ([_]usize{ trunk, branches, heads, kv_heads, group }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(scale);
        // the trunk's row tiles, then the branches': their rows' worth plus at most one partial tile each
        const blocks = (trunk + rows_per_block - 1) / rows_per_block + (rows - trunk + branches * (rows_per_block - 1)) / rows_per_block;
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(blocks), .y = @intCast(heads / heads_per_block) };
        try cuda.launch.launch(a.kernel, .{ .grid = grid, .block = .{ .x = threads }, .shared = smem }, s, &args);
    }
};

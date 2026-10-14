//! LFM2's prompt attention: prefill_attention.cu's flash kernel at head dim 64, causal over one sequence.
const cuda = @import("cuda");

const sym = "_ZN20tf_prefill_attention12pattn_kernelILi64ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
/// The instantiation pattn_kernel<64, 8, 4, 8>: eight warps as four query heads by two 16-row tiles.
const head_dim = 64;
const heads_per_block = 4;
const rows_per_block = 32;
const threads = 256;
const smem: u32 = 32768; // eight 32-key slots of 64 dims

pub const Attention = struct {
    mod: cuda.Module,
    pattn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Attention {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.prefill_attention);
        errdefer mod.unload();
        return .{ .mod = mod, .pattn = try mod.function(sym) };
    }

    pub fn unload(a: *Attention) void {
        a.mod.unload();
    }

    /// Causal GQA over one sequence: q [rows, H, D], k/v [rows, KV, D] bf16 -> out [rows, H, D] bf16.
    pub fn causal(a: *const Attention, s: cuda.Stream, q: u64, k: u64, v: u64, out: u64, rows: usize, heads: usize, kv_heads: usize, dim: usize, scale: f32) !void {
        const group = heads / kv_heads;
        if (dim != head_dim or group * kv_heads != heads or group % heads_per_block != 0) return error.UnsupportedShape;
        if (rows == 0) return;
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, v, out }) |p| args.add(p);
        // p0 0: the keys are the queries' own rows, k and v serve as the caches
        for ([_]usize{ 0, rows, heads, kv_heads, group }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(scale);
        const grid: cuda.launch.Dim3 = .{ .x = @intCast((rows + rows_per_block - 1) / rows_per_block), .y = @intCast(heads / heads_per_block) };
        try cuda.launch.launch(a.pattn, .{ .grid = grid, .block = .{ .x = threads }, .shared = smem }, s, &args);
    }
};

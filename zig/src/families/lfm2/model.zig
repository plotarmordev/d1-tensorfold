//! LFM2's prompt pass: token rows through the conv and attention layers, read at the last row's labels.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Weights = @import("weights.zig").Weights;
const Kernels = @import("kernels.zig").Kernels;
const Attention = @import("attention.zig").Attention;
const Linear = cuda.cublaslt.Linear;
const Allocator = std.mem.Allocator;

const workspace_bytes = 32 << 20;
const labels_most = 4096;

pub const Model = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    cfg: *const Config,
    w: *const Weights,
    k: Kernels,
    at: Attention,
    embed: cuda.embed_rows.EmbedRows,
    lt: cuda.cublaslt.Library,
    stream: cuda.Stream,
    rows_most: usize,
    scratch: cuda.DeviceBuffer,
    workspace: cuda.DeviceBuffer,
    host: cuda.HostBuffer, // ids, positions, prev, labels in; label logits and the logsumexp out
    linears: std.AutoHashMapUnmanaged([3]u64, Linear) = .empty,
    b: Buffers,

    const Buffers = struct { ids: u64, pos: u64, prev: u64, labels: u64, h: u64, x: u64, wide: u64, y: u64, q: u64, kk: u64, v: u64, a: u64, g: u64, logits: u64, lse: u64 };

    /// Scratch for prompts of up to `rows_most` rows.
    pub fn init(gpa: Allocator, d: *const cuda.Driver, cfg: *const Config, w: *const Weights, rows_most: usize) !Model {
        var m: Model = .{ .gpa = gpa, .d = d, .cfg = cfg, .w = w, .rows_most = rows_most, .k = try Kernels.load(d), .at = undefined, .embed = undefined, .lt = undefined, .stream = undefined, .scratch = undefined, .workspace = undefined, .host = undefined, .b = undefined };
        errdefer m.k.unload();
        m.at = try Attention.load(d);
        errdefer m.at.unload();
        m.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer m.embed.unload();
        m.lt = try cuda.cublaslt.Library.open();
        errdefer m.lt.close();
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();
        const r = rows_most;
        const hd = cfg.heads * cfg.dim;
        const kvd = cfg.kv_heads * cfg.dim;
        const wide = @max(3 * cfg.hidden, hd + 2 * kvd, 2 * cfg.ffn);
        const taps = cfg.taps - 1;
        // i32 index rows, then bf16 activations, then f32 logits (a vocabulary's worth for the logsumexp)
        const sizes = [_]usize{ 4 * r, 4 * r, 4 * r * taps, 4 * labels_most, 2 * r * cfg.hidden, 2 * r * cfg.hidden, 2 * r * wide, 2 * r * cfg.hidden, 2 * r * hd, 2 * r * kvd, 2 * r * kvd, 2 * r * hd, 2 * r * cfg.ffn, 4 * @max(cfg.vocab, labels_most), 4 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        m.scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer m.scratch.free();
        var at: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &at) |s, *p| {
            p.* = try m.scratch.at(off);
            off += std.mem.alignForward(usize, s, 256);
        }
        m.b = .{ .ids = at[0], .pos = at[1], .prev = at[2], .labels = at[3], .h = at[4], .x = at[5], .wide = at[6], .y = at[7], .q = at[8], .kk = at[9], .v = at[10], .a = at[11], .g = at[12], .logits = at[13], .lse = at[14] };
        m.workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes);
        errdefer m.workspace.free();
        m.host = try cuda.HostBuffer.alloc(d, 4 * (r * (2 + taps) + 2 * labels_most + 1));
        return m;
    }

    pub fn deinit(m: *Model) void {
        var it = m.linears.valueIterator();
        while (it.next()) |l| l.deinit();
        m.linears.deinit(m.gpa);
        m.host.free();
        m.workspace.free();
        m.scratch.free();
        m.stream.deinit();
        m.lt.close();
        m.embed.unload();
        m.at.unload();
        m.k.unload();
    }

    /// y [rows, n] = x [rows, k] . w [n, k]^T, a plan kept a shape.
    fn linear(m: *Model, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize, out: cuda.cublaslt.DataType) !void {
        const key = [3]u64{ rows, n, k | (@as(u64, @intCast(@intFromEnum(out))) << 48) };
        const got = try m.linears.getOrPut(m.gpa, key);
        if (!got.found_existing) got.value_ptr.* = Linear.init(&m.lt, rows, n, k, out, workspace_bytes) catch |e| {
            m.linears.removeByPtr(got.key_ptr);
            return e;
        };
        try got.value_ptr.run(x, w, y, m.workspace.ptr, workspace_bytes, m.stream.handle);
    }

    /// One logit per label at the row after `prompt` (f32, as model.logits); the vocabulary logsumexp when `mass`.
    pub fn score(m: *Model, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) !f64 {
        const c = m.cfg;
        const rows = prompt.len;
        if (rows == 0 or rows > m.rows_most) return error.PromptLength;
        if (labels.len > labels_most) return error.TooManyLabels;
        const taps = c.taps - 1;
        const host = m.host.slice(i32);
        const ids = host[0..rows];
        const pos = host[rows .. 2 * rows];
        const prev = host[2 * rows ..][0 .. rows * taps];
        const lab = host[rows * (2 + taps) ..][0..labels.len];
        for (prompt, ids, pos, 0..) |t, *i, *p, r| {
            i.* = @intCast(t);
            p.* = @intCast(r);
            for (0..taps) |j| prev[r * taps + j] = if (r > j) @intCast(r - j - 1) else -1; // newest first
        }
        for (labels, lab) |l, *x| x.* = @intCast(l);
        const s = m.stream;
        try m.put(m.b.ids, std.mem.sliceAsBytes(host[0..rows]));
        try m.put(m.b.pos, std.mem.sliceAsBytes(pos));
        try m.put(m.b.prev, std.mem.sliceAsBytes(prev));
        try m.put(m.b.labels, std.mem.sliceAsBytes(lab));
        try m.forward(rows);
        const last = m.b.x + 2 * (rows - 1) * c.hidden; // the final-normed row after the prompt
        try m.k.labelLogits(s, last, m.w.embed, m.b.labels, m.b.logits, labels.len, c.hidden);
        const out = m.host.slice(f32)[rows * (2 + taps) + labels_most ..][0 .. labels_most + 1];
        try m.get(m.b.logits, std.mem.sliceAsBytes(out[0..labels.len]));
        var lse: f64 = 0;
        if (mass) {
            try m.linear(last, m.w.embed, m.b.logits, 1, c.vocab, c.hidden, .f32);
            try m.k.logsumexp(s, m.b.logits, m.b.lse, c.vocab);
            try m.get(m.b.lse, std.mem.sliceAsBytes(out[labels_most..][0..1]));
        }
        try s.synchronize();
        for (logits, out[0..labels.len]) |*x, v| x.* = v;
        if (mass) lse = out[labels_most];
        return lse;
    }

    /// Pinned host bytes to a scratch address, on the model's stream.
    fn put(m: *Model, dst: u64, bytes: []const u8) !void {
        try m.scratch.uploadAsync(dst - m.scratch.ptr, bytes, m.stream.handle);
    }

    fn get(m: *Model, src: u64, out: []u8) !void {
        try m.scratch.downloadAsync(src - m.scratch.ptr, out, m.stream.handle);
    }

    /// The layers over the uploaded rows; leaves the final-normed last row in x.
    fn forward(m: *Model, rows: usize) !void {
        const c = m.cfg;
        const s = m.stream;
        const b = m.b;
        const hd = c.heads * c.dim;
        const kvd = c.kv_heads * c.dim;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dim)));
        try m.embed.bf16(s, b.ids, m.w.embed, b.h, rows, c.hidden);
        try m.k.addRms(s, b.h, null, m.w.layers[0].op_norm, b.x, rows, c.hidden, c.eps);
        for (m.w.layers, 0..) |l, i| {
            switch (l.kind) {
                .conv => {
                    try m.linear(b.x, l.proj, b.wide, rows, 3 * c.hidden, c.hidden, .bf16);
                    try m.k.shortConv(s, b.wide, l.conv, b.prev, b.a, rows, c.hidden, c.taps);
                    try m.linear(b.a, l.out, b.y, rows, c.hidden, c.hidden, .bf16);
                },
                .attention => {
                    try m.linear(b.x, l.proj, b.wide, rows, hd + 2 * kvd, c.hidden, .bf16);
                    try m.k.qkNormRope(s, b.wide, l.q_norm, l.k_norm, b.pos, b.q, b.kk, b.v, rows, c.heads, c.kv_heads, c.dim, c.theta, c.eps);
                    try m.at.causal(s, b.q, b.kk, b.v, b.a, rows, c.heads, c.kv_heads, c.dim, scale);
                    try m.linear(b.a, l.out, b.y, rows, c.hidden, hd, .bf16);
                },
            }
            try m.k.addRms(s, b.h, b.y, l.ffn_norm, b.x, rows, c.hidden, c.eps);
            try m.linear(b.x, l.up, b.wide, rows, 2 * c.ffn, c.hidden, .bf16);
            try m.k.siluMul(s, b.wide, b.g, rows, c.ffn);
            try m.linear(b.g, l.down, b.y, rows, c.hidden, c.ffn, .bf16);
            if (i + 1 < m.w.layers.len) try m.k.addRms(s, b.h, b.y, m.w.layers[i + 1].op_norm, b.x, rows, c.hidden, c.eps);
        }
        // the final norm at the last row alone, as the Python port normalizes only the rows it reads
        const at = 2 * (rows - 1) * c.hidden;
        try m.k.addRms(s, b.h + at, b.y + at, m.w.norm, b.x + at, 1, c.hidden, c.eps);
    }
};

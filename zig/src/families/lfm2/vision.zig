//! LFM2-VL's pictures on CUDA: the SigLIP2 NaFlex tower and the projector a tile at a time, as vision.py runs them.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const VisionWeights = @import("weights.zig").VisionWeights;
const image = @import("image.zig");
const Allocator = std.mem.Allocator;
const Linear = cuda.cublaslt.Linear;

const threads = 256; // lfm2_vision.cu's NT
const norm_most = 8 * threads; // NT * MAXE
const keys_most = 1024; // KMAX
const attend_rows = 8; // AR
const dim_most = 128; // DMAX
const patches_most = keys_most; // a 512-pixel tile, or a thumbnail of at most 256 tokens
const tiles_most = image.max_grid + 1; // max_tiles and the thumbnail
const workspace_bytes = 32 << 20;
const px_bytes = 2 * tiles_most * patches_most * image.patch_width; // a picture's bf16 patches, 256-byte aligned

/// What tf_lfm2v_bias applies after the bias: nothing, torch's tanh GELU, or its exact (erf) GELU.
pub const Act = enum(i32) { none = 0, gelu_tanh = 1, gelu = 2 };

pub const Kernels = struct {
    mod: cuda.Module,
    layer_norm_fn: cuda.Function,
    bias_fn: cuda.Function,
    split_fn: cuda.Function,
    attend_fn: cuda.Function,
    unshuffle_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.lfm2_vision);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .layer_norm_fn = try mod.function("tf_lfm2v_layer_norm"),
            .bias_fn = try mod.function("tf_lfm2v_bias"),
            .split_fn = try mod.function("tf_lfm2v_split_heads"),
            .attend_fn = try mod.function("tf_lfm2v_attend"),
            .unshuffle_fn = try mod.function("tf_lfm2v_unshuffle"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// y = LayerNorm(x) * w + b over rows up to 2048 wide.
    pub fn layerNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize, eps: f32) !void {
        if (width > norm_most) return error.RowShape;
        var a: cuda.Args = .{};
        for ([_]u64{ x, w, b, y }) |p| a.add(p);
        a.add(int(width));
        a.add(eps);
        try go(k.layer_norm_fn, s, rows, 1, &a);
    }

    /// o [rows, n] = bf16(acc + b) through `act`, then plus r when given (r may be o).
    pub fn bias(k: *const Kernels, s: cuda.Stream, acc: u64, b: u64, r: ?u64, o: u64, rows: usize, n: usize, act: Act) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ acc, b, r orelse 0, o }) |p| a.add(p);
        a.add(int(n));
        a.add(@backingInt(act));
        try go(k.bias_fn, s, rows, blocks(n, threads), &a);
    }

    /// qkv [rows, 3 H D] -> q and k [H][rows][D], heads `hs` elements apart.
    pub fn splitHeads(k: *const Kernels, s: cuda.Stream, qkv: u64, q: u64, kk: u64, rows: usize, heads: usize, dim: usize, hs: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ qkv, q, kk }) |p| a.add(p);
        for ([_]usize{ heads, dim }) |v| a.add(int(v));
        a.add(@as(i64, @intCast(hs)));
        try go(k.split_fn, s, rows, 1, &a);
    }

    /// a [rows, H D] = softmax(scores * scale) . v in f32: scores [H][rows][rows] ss apart, v in qkv; rows <= 1024.
    pub fn attend(k: *const Kernels, s: cuda.Stream, scores: u64, qkv: u64, a_: u64, rows: usize, heads: usize, dim: usize, ss: usize, scale: f32) !void {
        if (rows > keys_most or dim > dim_most) return error.RowShape;
        var a: cuda.Args = .{};
        for ([_]u64{ scores, qkv, a_ }) |p| a.add(p);
        for ([_]usize{ rows, heads, dim }) |v| a.add(int(v));
        a.add(@as(i64, @intCast(ss)));
        a.add(scale);
        try go(k.attend_fn, s, blocks(rows, attend_rows), heads, &a);
    }

    /// y [h/2 * w/2, 4c] = the projector's pixel unshuffle of x [h * w, c], an even grid.
    pub fn unshuffle(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, h: usize, w: usize, c: usize) !void {
        if (h % 2 != 0 or w % 2 != 0) return error.RowShape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(int(w));
        a.add(int(c));
        try go(k.unshuffle_fn, s, h * w / 4, 1, &a);
    }
};

fn int(x: usize) i32 {
    return @intCast(x);
}

fn blocks(n: usize, by: usize) usize {
    return (n + by - 1) / by;
}

fn go(f: cuda.Function, s: cuda.Stream, x: usize, y: usize, a: *cuda.Args) !void {
    if (x == 0) return;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x), .y = @intCast(y), .z = 1 }, .block = .{ .x = threads } }, s, a);
}

/// torch Linears through cuBLASLt: a plan kept a shape, one workspace.
pub const Gemms = struct {
    gpa: Allocator,
    lt: cuda.cublaslt.Library,
    workspace: cuda.DeviceBuffer,
    plans: std.AutoHashMapUnmanaged([3]u64, Linear) = .empty,

    pub fn init(gpa: Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try cuda.cublaslt.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y [rows, n] = x [rows, k] . w [n, k]^T, f32 accumulation, `out` rows.
    pub fn run(g: *Gemms, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize, out: cuda.cublaslt.DataType) !void {
        const key = [3]u64{ rows, n, k | (@as(u64, @intCast(@backingInt(out))) << 48) };
        const got = try g.plans.getOrPut(g.gpa, key);
        if (!got.found_existing) got.value_ptr.* = Linear.init(&g.lt, rows, n, k, out, workspace_bytes) catch |e| {
            g.plans.removeByPtr(got.key_ptr);
            return e;
        };
        try got.value_ptr.run(x, w, y, g.workspace.ptr, workspace_bytes, s.handle);
    }
};

/// Elements between heads, each head 256-byte aligned as cuBLASLt's default plans assume its operands.
pub const Strides = struct {
    hs: usize, // q and k: bf16
    ss: usize, // scores: f32

    pub fn of(rows: usize, dim: usize) Strides {
        return .{ .hs = std.mem.alignForward(usize, rows * dim, 128), .ss = std.mem.alignForward(usize, rows * rows, 64) };
    }
};

/// One tile's attention scratch, laid out by `Strides.of(rows, dim)`.
pub const Heads = struct { q: u64, k: u64, scores: u64 };

/// a [rows, H D] = SDPA over one tile's rows (no mask), from qkv [rows, 3 H D] with its biases added.
pub fn attention(k: *const Kernels, g: *Gemms, s: cuda.Stream, b: Heads, qkv: u64, a: u64, rows: usize, heads: usize, dim: usize) !void {
    const st = Strides.of(rows, dim);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
    try k.splitHeads(s, qkv, b.q, b.k, rows, heads, dim, st.hs);
    for (0..heads) |h| try g.run(s, b.q + 2 * h * st.hs, b.k + 2 * h * st.hs, b.scores + 4 * h * st.ss, rows, rows, dim, .f32);
    try k.attend(s, b.scores, qkv, a, rows, heads, dim, st.ss, scale);
}

pub const Vision = struct {
    c: config.Vision,
    text_hidden: usize,
    w: *const VisionWeights,
    k: Kernels,
    g: Gemms,
    position: []f32, // the position table [side, side, hidden] on the host, resized for each tile's grid
    scratch: cuda.DeviceBuffer,
    staging: cuda.HostBuffer, // pinned: a picture's bf16 patches, then its tiles' position rows
    b: Buffers,

    const Buffers = struct { px: u64, pos: u64, h: u64, x: u64, wide: u64, a: u64, acc: u64, heads: Heads };

    /// Kernels, cuBLASLt and scratch for one picture of up to 11 tiles of 1024 patches.
    pub fn init(gpa: Allocator, d: *const cuda.Driver, cfg: *const config.Config, w: *const VisionWeights) !Vision {
        const c = cfg.vision orelse return error.NoVisionTower;
        const dim = c.hidden / c.heads;
        const side = std.math.sqrt(c.num_patches);
        if (dim * c.heads != c.hidden or dim > dim_most or c.hidden > norm_most or c.downsample != 2 or side * side != c.num_patches or c.channels * c.patch * c.patch != image.patch_width) return error.VisionShape;
        var k = try Kernels.load(d);
        errdefer k.unload();
        var g = try Gemms.init(gpa, d);
        errdefer g.deinit();
        const position = try gpa.alloc(f32, c.num_patches * c.hidden);
        errdefer gpa.free(position);
        const table = try gpa.alloc(u16, position.len);
        defer gpa.free(table);
        try d.check(d.api.cuMemcpyDtoH_v2(table.ptr, w.position, 2 * table.len), "cuMemcpyDtoH");
        for (position, table) |*p, t| p.* = @bitCast(@as(u32, t) << 16);
        const n = patches_most;
        const st = Strides.of(n, dim);
        const merged = 4 * c.hidden;
        const staged = [_]usize{ px_bytes, 2 * tiles_most * n * c.hidden };
        // bf16 rows then the f32 Linear outputs, then the attention's q and k heads and f32 scores
        const sizes = staged ++ [_]usize{
            2 * n * c.hidden,
            2 * n * @max(c.hidden, c.projector_hidden / 4),
            2 * n * @max(3 * c.hidden, c.ffn, merged / 4),
            2 * n * c.hidden,
            4 * n * @max(3 * c.hidden, c.ffn, @max(c.projector_hidden, cfg.hidden) / 4),
            2 * c.heads * st.hs,
            2 * c.heads * st.hs,
            4 * c.heads * st.ss,
        };
        var total: usize = 0;
        for (sizes) |sz| total += std.mem.alignForward(usize, sz, 256);
        var scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer scratch.free();
        var at: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &at) |sz, *p| {
            p.* = try scratch.at(off);
            off += std.mem.alignForward(usize, sz, 256);
        }
        return .{
            .c = c,
            .text_hidden = cfg.hidden,
            .w = w,
            .k = k,
            .g = g,
            .position = position,
            .scratch = scratch,
            .staging = try cuda.HostBuffer.alloc(d, px_bytes + staged[1]),
            .b = .{ .px = at[0], .pos = at[1], .h = at[2], .x = at[3], .wide = at[4], .a = at[5], .acc = at[6], .heads = .{ .q = at[7], .k = at[8], .scores = at[9] } },
        };
    }

    pub fn deinit(v: *Vision) void {
        v.staging.free();
        v.scratch.free();
        v.g.gpa.free(v.position);
        v.g.deinit();
        v.k.unload();
    }

    /// One picture's features, bf16 [tokens, text hidden] at `out`, tiles row by row then the thumbnail; the rows.
    pub fn features(v: *Vision, s: cuda.Stream, picture: *const image.Picture, out: u64) !usize {
        const c = v.c;
        if (picture.tiles.len > tiles_most) return error.TileShape;
        var n: usize = 0;
        for (picture.tiles) |t| {
            if (t.h * t.w > patches_most or t.h % 2 != 0 or t.w % 2 != 0) return error.TileShape;
            n += t.h * t.w;
        }
        try s.synchronize(); // the staging rows may still feed the last picture's copies
        const host = v.staging.slice(u16);
        const px = host[0 .. n * image.patch_width];
        for (px, picture.patches[0..px.len]) |*o, p| o.* = bf16(p);
        const pos = host[px_bytes / 2 ..][0 .. n * c.hidden];
        var at: usize = 0;
        for (picture.tiles, 0..) |t, i| {
            const rows = pos[at * c.hidden ..][0 .. t.h * t.w * c.hidden];
            if (i > 0 and std.meta.eql(t, picture.tiles[i - 1])) @memcpy(rows, pos[(at - t.h * t.w) * c.hidden ..][0..rows.len]) else try v.resizePositions(t, rows);
            at += t.h * t.w;
        }
        try v.scratch.uploadAsync(v.b.px - v.scratch.ptr, std.mem.sliceAsBytes(px), s.handle);
        try v.scratch.uploadAsync(v.b.pos - v.scratch.ptr, std.mem.sliceAsBytes(pos), s.handle);
        var row: usize = 0;
        at = 0;
        for (picture.tiles) |t| {
            try v.tile(s, t, v.b.px + 2 * at * image.patch_width, v.b.pos + 2 * at * c.hidden, out + 2 * row * v.text_hidden);
            at += t.h * t.w;
            row += t.h * t.w / 4;
        }
        return row;
    }

    /// The tower and projector over one tile's patches and position rows: its tokens' features at `out`.
    fn tile(v: *Vision, s: cuda.Stream, t: image.Grid, px: u64, pos: u64, out: u64) !void {
        const c = v.c;
        const b = v.b;
        const k = &v.k;
        const w = v.w;
        const n = t.h * t.w;
        const hd = c.hidden;
        try v.g.run(s, px, w.patch, b.acc, n, hd, image.patch_width, .f32);
        try k.bias(s, b.acc, w.patch_bias, pos, b.h, n, hd, .none);
        for (w.layers) |l| {
            try k.layerNorm(s, b.h, l.norm1, l.norm1_bias, b.x, n, hd, c.eps);
            try v.g.run(s, b.x, l.qkv, b.acc, n, 3 * hd, hd, .f32);
            try k.bias(s, b.acc, l.qkv_bias, null, b.wide, n, 3 * hd, .none);
            try attention(k, &v.g, s, b.heads, b.wide, b.a, n, c.heads, hd / c.heads);
            try v.g.run(s, b.a, l.out, b.acc, n, hd, hd, .f32);
            try k.bias(s, b.acc, l.out_bias, b.h, b.h, n, hd, .none);
            try k.layerNorm(s, b.h, l.norm2, l.norm2_bias, b.x, n, hd, c.eps);
            try v.g.run(s, b.x, l.fc1, b.acc, n, c.ffn, hd, .f32);
            try k.bias(s, b.acc, l.fc1_bias, null, b.wide, n, c.ffn, .gelu_tanh);
            try v.g.run(s, b.wide, l.fc2, b.acc, n, hd, c.ffn, .f32);
            try k.bias(s, b.acc, l.fc2_bias, b.h, b.h, n, hd, .none);
        }
        try k.layerNorm(s, b.h, w.post_norm, w.post_norm_bias, b.x, n, hd, c.eps);
        const tokens = n / 4;
        try k.unshuffle(s, b.x, b.wide, t.h, t.w, hd);
        try v.g.run(s, b.wide, w.linear_1, b.acc, tokens, c.projector_hidden, 4 * hd, .f32);
        try k.bias(s, b.acc, w.linear_1_bias, null, b.x, tokens, c.projector_hidden, .gelu);
        try v.g.run(s, b.x, w.linear_2, b.acc, tokens, v.text_hidden, c.projector_hidden, .f32);
        try k.bias(s, b.acc, w.linear_2_bias, null, out, tokens, v.text_hidden, .none);
    }

    /// The position table resized to grid `t` (Siglip2VisionEmbeddings.resize_positional_embeddings), bf16 rows.
    fn resizePositions(v: *Vision, t: image.Grid, out: []u16) !void {
        const side = std.math.sqrt(v.c.num_patches);
        const gpa = v.g.gpa;
        const resized = try gpa.alloc(f32, out.len);
        defer gpa.free(resized);
        try resize(gpa, v.position, side, v.c.hidden, t.h, t.w, resized);
        for (out, resized) |*o, x| o.* = bf16(x);
    }
};

/// cvt.rn.bf16.f32 of a finite value.
fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// torch's antialiased bilinear weights from `from` samples to `to` (align_corners false), [to][from].
fn weights(a: Allocator, from: usize, to: usize) ![]f64 {
    const out = try a.alloc(f64, to * from);
    @memset(out, 0);
    const scale = @as(f64, @floatFromInt(from)) / @as(f64, @floatFromInt(to));
    const support = @max(scale, 1.0);
    const inv = 1.0 / @max(scale, 1.0);
    for (0..to) |i| {
        const center = scale * (@as(f64, @floatFromInt(i)) + 0.5);
        const lo: usize = @intFromFloat(@max(center - support + 0.5, 0));
        const hi: usize = @min(@as(usize, @intFromFloat(center + support + 0.5)), from);
        const row = out[i * from ..][0..from];
        var total: f64 = 0;
        for (lo..hi) |j| {
            row[j] = @max(0, 1 - @abs((@as(f64, @floatFromInt(j)) - center + 0.5) * inv));
            total += row[j];
        }
        if (total != 0) for (row[lo..hi]) |*x| {
            x.* /= total;
        };
    }
    return out;
}

/// F.interpolate(bilinear, align_corners=False, antialias=True) of a [side, side, c] table to [h, w, c].
fn resize(a: Allocator, table: []const f32, side: usize, c: usize, h: usize, w: usize, out: []f32) !void {
    const wy = try weights(a, side, h);
    defer a.free(wy);
    const wx = try weights(a, side, w);
    defer a.free(wx);
    const across = try a.alloc(f64, side * w * c);
    defer a.free(across);
    @memset(across, 0);
    for (0..side) |i| for (0..w) |x| for (0..side) |j| {
        const f = wx[x * side + j];
        for (across[(i * w + x) * c ..][0..c], table[(i * side + j) * c ..][0..c]) |*o, v| o.* += f * v;
    };
    const sum = try a.alloc(f64, c);
    defer a.free(sum);
    for (0..h) |y| for (0..w) |x| {
        @memset(sum, 0);
        for (0..side) |i| {
            const f = wy[y * side + i];
            for (sum, across[(i * w + x) * c ..][0..c]) |*o, v| o.* += f * v;
        }
        for (out[(y * w + x) * c ..][0..c], sum) |*o, v| o.* = @floatCast(v);
    };
}

test "position rows resize as torch's antialiased bilinear interpolate" {
    const a = std.testing.allocator;
    const side = 16;
    const c = 3;
    var table: [side * side * c]f32 = undefined;
    for (0..side) |i| for (0..side) |j| for (0..c) |k| {
        table[(i * side + j) * c + k] = @floatCast(@sin(0.37 * @as(f64, @floatFromInt(i)) + 0.11 * @as(f64, @floatFromInt(j)) + 0.05 * @as(f64, @floatFromInt(k))));
    };
    // torch 2.13 F.interpolate on the CPU: (h, w) then (y, x, channel, value) picks
    const Pick = struct { y: usize, x: usize, k: usize, v: f32 };
    const cases = [_]struct { h: usize, w: usize, picks: [4]Pick }{
        .{ .h = 32, .w = 32, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.0 }, .{ .y = 31, .x = 31, .k = 2, .v = 0.8504366278648376 }, .{ .y = 16, .x = 10, .k = 1, .v = -0.290686696767807 }, .{ .y = 1, .x = 30, .k = 0, .v = 0.9759851694107056 } } },
        .{ .h = 8, .w = 34, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.25130489468574524 }, .{ .y = 7, .x = 33, .k = 2, .v = 0.6617922186851501 }, .{ .y = 4, .x = 11, .k = 1, .v = -0.5310338735580444 }, .{ .y = 1, .x = 32, .k = 0, .v = 0.5272247195243835 } } },
        .{ .h = 22, .w = 14, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.018296388909220695 }, .{ .y = 21, .x = 13, .k = 2, .v = 0.8399547338485718 }, .{ .y = 11, .x = 4, .k = 1, .v = -0.31715455651283264 }, .{ .y = 1, .x = 12, .k = 0, .v = 0.9701255559921265 } } },
        .{ .h = 4, .w = 100, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.559954822063446 }, .{ .y = 3, .x = 99, .k = 2, .v = 0.27992990612983704 }, .{ .y = 2, .x = 33, .k = 1, .v = -0.6748089790344238 }, .{ .y = 1, .x = 98, .k = 0, .v = -0.4268251955509186 } } },
    };
    for (cases) |case| {
        const out = try a.alloc(f32, case.h * case.w * c);
        defer a.free(out);
        try resize(a, &table, side, c, case.h, case.w, out);
        for (case.picks) |p| try std.testing.expectApproxEqAbs(p.v, out[(p.y * case.w + p.x) * c + p.k], 1e-6);
    }
}

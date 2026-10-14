//! LFM2's prompt attention against a float64 causal softmax of the same bf16 rows, and its bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const lfm2 = @import("lfm2");
const check = @import("check.zig");

const heads = 32;
const kv_heads = 8;
const dim = 64;
const scale: f32 = 0.125; // 1 / sqrt(dim)
const counts = [_]usize{ 1, 7, 33, 300, 4096 };
const most = counts[counts.len - 1];
const prefix = 300; // rows checked against the host in full; past it every 61st and the last

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn float(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Seeded bf16 values in [-2, 2): scores at scale 1/8 span a few units, so the softmax is neither flat nor one-hot.
fn values(a: std.mem.Allocator, rng: std.Random, n: usize) !struct { bits: []u16, vals: []f32 } {
    const bits = try a.alloc(u16, n);
    const vals = try a.alloc(f32, n);
    for (bits, vals) |*b, *v| {
        b.* = bf16(rng.float(f32) * 4 - 2);
        v.* = float(b.*);
    }
    return .{ .bits = bits, .vals = vals };
}

/// Row `row` in float64 and its bound: one bf16 rounding of the output, two of each probability (ex2.approx, f32 sums).
fn reference(q: []const f32, k: []const f32, v: []const f32, row: usize, p: []f64, want: []f64, bound: []f64) void {
    for (0..heads) |h| {
        const qh = q[(row * heads + h) * dim ..][0..dim];
        const kh = h / (heads / kv_heads);
        var top = -std.math.inf(f64);
        for (p[0 .. row + 1], 0..) |*s, j| {
            var dot: f64 = 0;
            for (qh, k[(j * kv_heads + kh) * dim ..][0..dim]) |x, y| dot += @as(f64, x) * y;
            s.* = dot * scale;
            top = @max(top, s.*);
        }
        var sum: f64 = 0;
        for (p[0 .. row + 1]) |*s| {
            s.* = @exp(s.* - top);
            sum += s.*;
        }
        const w = want[h * dim ..][0..dim];
        const b = bound[h * dim ..][0..dim];
        @memset(w, 0);
        @memset(b, 0);
        for (p[0 .. row + 1], 0..) |s, j| {
            for (w, b, v[(j * kv_heads + kh) * dim ..][0..dim]) |*o, *m, x| {
                o.* += s * x;
                m.* += s * @abs(x);
            }
        }
        for (w, b) |*o, *m| {
            o.* /= sum;
            m.* = (@abs(o.*) + 2 * m.* / sum) / 256;
        }
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var attn = try lfm2.attention.Attention.load(gpu.d);
    defer attn.unload();
    const refused = if (attn.causal(s, 0, 0, 0, 0, 1, heads, kv_heads, 128, scale)) |_| false else |e| e == error.UnsupportedShape;
    try check.expect(refused, "lfm2-attention: head dim 128 is refused", .{});

    var prng = std.Random.DefaultPrng.init(64);
    const q = try values(a, prng.random(), most * heads * dim);
    const k = try values(a, prng.random(), most * kv_heads * dim);
    const v = try values(a, prng.random(), most * kv_heads * dim);
    var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q.bits));
    defer dq.free();
    var dk = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(k.bits));
    defer dk.free();
    var dv = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(v.bits));
    defer dv.free();
    var outs: [counts.len]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (outs[0..made]) |*o| o.free();
    for (counts, &outs) |m, *o| {
        o.* = try cuda.DeviceBuffer.alloc(gpu.d, m * heads * dim * 2);
        made += 1;
        try o.fill8(0xff, s.handle); // a row the kernel misses reads NaN
        try attn.causal(s, dq.ptr, dk.ptr, dv.ptr, o.ptr, m, heads, kv_heads, dim, scale);
    }
    try s.synchronize();

    const wide = try a.alloc(u16, most * heads * dim);
    try outs[counts.len - 1].download(0, std.mem.sliceAsBytes(wide));
    const nonfinite = for (wide, 0..) |g, i| {
        if (!std.math.isFinite(float(g))) break i;
    } else null;
    try check.expect(nonfinite == null, "lfm2-attention: every output finite (first not at element {?d})", .{nonfinite});
    const p = try a.alloc(f64, most);
    var want: [heads * dim]f64 = undefined;
    var bound: [heads * dim]f64 = undefined;
    var worst: f64 = 0;
    var err_most: f64 = 0;
    var failed: usize = 0;
    var checked: usize = 0;
    for (0..most) |row| {
        if (row >= prefix and row % 61 != 0 and row != most - 1) continue;
        reference(q.vals, k.vals, v.vals, row, p, &want, &bound);
        for (wide[row * heads * dim ..][0 .. heads * dim], want, bound) |g, w, b| {
            const err = @abs(float(g) - w);
            if (!(err <= b)) failed += 1;
            err_most = @max(err_most, err);
            worst = @max(worst, err / b);
        }
        checked += 1;
    }
    const past = "lfm2-attention: {d} elements of {d} rows past bound (worst {d:.3}, max abs {e:.3})";
    try check.expect(failed == 0, past, .{ failed, checked, worst, err_most });
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, o| {
        const got = try a.alloc(u8, m * heads * dim * 2);
        try o.download(0, got);
        try check.sameBytes("lfm2-attention prefix", got, std.mem.sliceAsBytes(wide[0 .. m * heads * dim]));
    }
    const said = "lfm2-attention: H {d}, KV {d}, D {d}, {d} of {d} rows within bound (worst {d:.3}, max abs {e:.3})";
    check.pass(said ++ ", every shorter launch's bits equal its prefix", .{ heads, kv_heads, dim, checked, most, worst, err_most });
}

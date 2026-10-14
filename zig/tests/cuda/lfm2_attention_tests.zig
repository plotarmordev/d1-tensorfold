//! LFM2's prompt attention against a float64 softmax of the same bf16 rows: one causal sequence, then packed trees.
const std = @import("std");
const cuda = @import("cuda");
const lfm2 = @import("lfm2");
const check = @import("check.zig");

const heads = 32;
const kv_heads = 8;
const dim = 64;
const width = heads * dim; // a q or out row's values
const kv_width = kv_heads * dim;
const scale: f32 = 0.125; // 1 / sqrt(dim)
const counts = [_]usize{ 1, 7, 33, 300, 4096 };
const most = counts[counts.len - 1];
const trunks = [_]usize{ 0, 1, 300, 3400 };
const fans = [_]usize{ 1, 3, 64 };
const longest = 60; // a branch's most rows
const cheap = 400; // rows over at most this many keys are all checked against the host

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn float(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

const Values = struct { bits: []u16, vals: []f32 };

/// Seeded bf16 values in [-2, 2): scores at scale 1/8 span a few units, so the softmax is neither flat nor one-hot.
fn values(a: std.mem.Allocator, rng: std.Random, n: usize) !Values {
    const bits = try a.alloc(u16, n);
    const vals = try a.alloc(f32, n);
    for (bits, vals) |*b, *v| {
        b.* = bf16(rng.float(f32) * 4 - 2);
        v.* = float(b.*);
    }
    return .{ .bits = bits, .vals = vals };
}

/// The keys a query row sees: rows [0, shared), then its own run's rows [own, row].
const Keys = struct {
    shared: usize = 0,
    own: usize = 0,

    fn count(ks: Keys, row: usize) usize {
        return ks.shared + row + 1 - ks.own;
    }

    fn at(ks: Keys, j: usize) usize {
        return if (j < ks.shared) j else ks.own + j - ks.shared;
    }
};

/// Rows checked against the host: every cheap one, then every 61st and the first and last of its run [first, end).
fn sampled(row: usize, keys: Keys, first: usize, end: usize) bool {
    return keys.count(row) <= cheap or row % 61 == 0 or row == first or row + 1 == end;
}

/// Sampled elements against their bounds: failures stick, the maxima only report.
const Tally = struct {
    worst: f64 = 0,
    err_most: f64 = 0,
    failed: usize = 0,
    rows: usize = 0,

    fn row(t: *Tally, got: []const u16, want: []const f64, bound: []const f64) void {
        for (got, want, bound) |g, w, b| {
            const err = @abs(float(g) - w);
            if (!(err <= b)) t.failed += 1;
            t.err_most = @max(t.err_most, err);
            t.worst = @max(t.worst, err / b);
        }
        t.rows += 1;
    }

    fn expect(t: Tally, what: []const u8) !void {
        const past = "{s}: {d} elements of {d} rows past bound (worst {d:.3}, max abs {e:.3})";
        try check.expect(t.failed == 0, past, .{ what, t.failed, t.rows, t.worst, t.err_most });
    }
};

fn finite(what: []const u8, out: []const u16) !void {
    const bad = for (out, 0..) |g, i| {
        if (!std.math.isFinite(float(g))) break i;
    } else null;
    try check.expect(bad == null, "{s}: every output finite (first not at element {?d})", .{ what, bad });
}

/// A case's q, k and v rows on the host and the device.
const Rows = struct {
    q: Values,
    k: Values,
    v: Values,
    dq: u64,
    dk: u64,
    dv: u64,
};

const Rig = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    attn: *const lfm2.attention.Attention,
    rng: std.Random,
    p: []f64, // the reference's probabilities, a key each
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn keep(r: *Rig, b: cuda.DeviceBuffer) !cuda.DeviceBuffer {
        try r.bufs.append(r.gpu.gpa, b);
        return b;
    }

    fn from(r: *Rig, bytes: []const u8) !cuda.DeviceBuffer {
        return r.keep(try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes));
    }

    fn out(r: *Rig, len: usize) !cuda.DeviceBuffer {
        const b = try r.keep(try cuda.DeviceBuffer.alloc(r.gpu.d, len));
        try b.fill8(0xff, r.s.handle); // a row the kernel misses reads NaN
        return b;
    }

    /// Rows [first, first + n) of a [*, width] bf16 buffer, once the stream's work has landed.
    fn back(r: *Rig, a: std.mem.Allocator, b: cuda.DeviceBuffer, first: usize, n: usize) ![]u16 {
        const o = try a.alloc(u16, n * width);
        try r.s.synchronize();
        try b.download(first * width * 2, std.mem.sliceAsBytes(o));
        return o;
    }

    fn release(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }

    fn inputs(r: *Rig, a: std.mem.Allocator, rows: usize) !Rows {
        const q = try values(a, r.rng, rows * width);
        const k = try values(a, r.rng, rows * kv_width);
        const v = try values(a, r.rng, rows * kv_width);
        const dq = try r.from(std.mem.sliceAsBytes(q.bits));
        const dk = try r.from(std.mem.sliceAsBytes(k.bits));
        const dv = try r.from(std.mem.sliceAsBytes(v.bits));
        return .{ .q = q, .k = k, .v = v, .dq = dq.ptr, .dk = dk.ptr, .dv = dv.ptr };
    }

    /// Row `row` of `got` against float64 over `keys`, bounded by one bf16 rounding of it and two of each probability.
    fn compare(r: *Rig, t: *Tally, in: Rows, got: []const u16, row: usize, keys: Keys) void {
        var want: [width]f64 = undefined;
        var bound: [width]f64 = undefined;
        const p = r.p[0..keys.count(row)];
        for (0..heads) |h| {
            const qh = in.q.vals[(row * heads + h) * dim ..][0..dim];
            const kh = h / (heads / kv_heads);
            var top = -std.math.inf(f64);
            for (p, 0..) |*s, j| {
                var dot: f64 = 0;
                for (qh, in.k.vals[(keys.at(j) * kv_heads + kh) * dim ..][0..dim]) |x, y| dot += @as(f64, x) * y;
                s.* = dot * scale;
                top = @max(top, s.*);
            }
            var sum: f64 = 0;
            for (p) |*s| {
                s.* = @exp(s.* - top);
                sum += s.*;
            }
            const w = want[h * dim ..][0..dim];
            const b = bound[h * dim ..][0..dim];
            @memset(w, 0);
            @memset(b, 0);
            for (p, 0..) |s, j| {
                for (w, b, in.v.vals[(keys.at(j) * kv_heads + kh) * dim ..][0..dim]) |*o, *m, x| {
                    o.* += s * x;
                    m.* += s * @abs(x);
                }
            }
            for (w, b) |*o, *m| {
                o.* /= sum;
                m.* = (@abs(o.*) + 2 * m.* / sum) / 256;
            }
        }
        t.row(got[row * width ..][0..width], &want, &bound);
    }
};

/// One sequence at every row count: the longest against the host, the shorter ones its prefix bit for bit.
fn causalCase(r: *Rig, t: *Tally) !void {
    var arena: std.heap.ArenaAllocator = .init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const in = try r.inputs(a, most);
    var outs: [counts.len]cuda.DeviceBuffer = undefined;
    for (counts, &outs) |m, *o| {
        o.* = try r.out(m * width * 2);
        try r.attn.causal(r.s, in.dq, in.dk, in.dv, o.ptr, m, heads, kv_heads, dim, scale);
    }
    const wide = try r.back(a, outs[counts.len - 1], 0, most);
    try finite("lfm2-attention causal", wide);
    for (0..most) |row| if (sampled(row, .{}, 0, most)) r.compare(t, in, wide, row, .{});
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, o| {
        const got = try r.back(a, o, 0, m);
        try check.sameBytes("lfm2-attention causal prefix", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(wide[0 .. m * width]));
    }
}

/// A trunk and `fan` branches of 1..longest rows against the host, and each branch against causal over trunk ++ it.
fn treeCase(r: *Rig, t: *Tally, trunk: usize, fan: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const cu = try a.alloc(i32, fan + 1);
    cu[0] = 0;
    for (cu[1..], cu[0..fan]) |*c, prev| c.* = prev + r.rng.intRangeAtMost(i32, 1, longest);
    const rows = trunk + @as(usize, @intCast(cu[fan]));
    const in = try r.inputs(a, rows);
    const dcu = try r.from(std.mem.sliceAsBytes(cu));
    const dout = try r.out(rows * width * 2);
    try r.attn.tree(r.s, in.dq, in.dk, in.dv, dout.ptr, trunk, dcu.ptr, fan, rows, heads, kv_heads, dim, scale);
    const got = try r.back(a, dout, 0, rows);
    var name: [64]u8 = undefined;
    const what = try std.fmt.bufPrint(&name, "lfm2-attention tree, trunk {d}, {d} branches", .{ trunk, fan });
    try finite(what, got);
    for (0..trunk) |row| if (sampled(row, .{}, 0, trunk)) r.compare(t, in, got, row, .{});

    const joint = trunk + longest;
    const jq = try r.out(joint * width * 2);
    const jk = try r.out(joint * kv_width * 2);
    const jv = try r.out(joint * kv_width * 2);
    const jout = try r.out(joint * width * 2);
    try jq.copyFrom(0, in.dq, trunk * width * 2, r.s.handle);
    try jk.copyFrom(0, in.dk, trunk * kv_width * 2, r.s.handle);
    try jv.copyFrom(0, in.dv, trunk * kv_width * 2, r.s.handle);
    for (cu[0..fan], cu[1..]) |c0, c1| {
        const first = trunk + @as(usize, @intCast(c0));
        const end = trunk + @as(usize, @intCast(c1));
        const keys: Keys = .{ .shared = trunk, .own = first };
        for (first..end) |row| if (sampled(row, keys, first, end)) r.compare(t, in, got, row, keys);
        const n = end - first;
        try jq.copyFrom(trunk * width * 2, in.dq + first * width * 2, n * width * 2, r.s.handle);
        try jk.copyFrom(trunk * kv_width * 2, in.dk + first * kv_width * 2, n * kv_width * 2, r.s.handle);
        try jv.copyFrom(trunk * kv_width * 2, in.dv + first * kv_width * 2, n * kv_width * 2, r.s.handle);
        try r.attn.causal(r.s, jq.ptr, jk.ptr, jv.ptr, jout.ptr, trunk + n, heads, kv_heads, dim, scale);
        const from = if (c0 == 0) 0 else trunk; // the trunk's rows once, beside the first branch
        const alone = try r.back(a, jout, from, trunk + n - from);
        const shared = (trunk - from) * width;
        try check.sameBytes(what, std.mem.sliceAsBytes(alone[0..shared]), std.mem.sliceAsBytes(got[0..shared]));
        try check.sameBytes(what, std.mem.sliceAsBytes(alone[shared..]), std.mem.sliceAsBytes(got[first * width .. end * width]));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var attn = try lfm2.attention.Attention.load(gpu.d);
    defer attn.unload();
    const refused = if (attn.causal(s, 0, 0, 0, 0, 1, heads, kv_heads, 128, scale)) |_| false else |e| e == error.UnsupportedShape;
    try check.expect(refused, "lfm2-attention: head dim 128 is refused", .{});

    var prng = std.Random.DefaultPrng.init(64);
    const p = try gpu.gpa.alloc(f64, @max(most, trunks[trunks.len - 1] + longest));
    defer gpu.gpa.free(p);
    var r: Rig = .{ .gpu = gpu, .s = s, .attn = &attn, .rng = prng.random(), .p = p };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var causal: Tally = .{};
    try causalCase(&r, &causal);
    try causal.expect("lfm2-attention causal");
    var tree: Tally = .{};
    for (trunks) |trunk| for (fans) |fan| try treeCase(&r, &tree, trunk, fan);
    try tree.expect("lfm2-attention tree");
    const said = "lfm2-attention: causal {d} rows (worst {d:.3}, max abs {e:.3}), trees {d} rows (worst {d:.3}, max abs {e:.3})";
    check.pass(said ++ " within bound; shorter launches, trunks and branches bit-equal to causal", .{ causal.rows, causal.worst, causal.err_most, tree.rows, tree.worst, tree.err_most });
}

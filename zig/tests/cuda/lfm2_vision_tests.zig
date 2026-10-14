//! LFM2-VL's vision kernels against host references, then pictures' features against the Python port's CPU goldens.

const std = @import("std");
const cuda = @import("cuda");
const lfm2 = @import("lfm2");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const gm = nemotron.glue_math;
const bf = gm.bf16ToF32;
const vision = lfm2.vision;

const hidden = 1152;
const heads = 16;
const dim = 72;
const ffn = 4304;
const eps: f32 = 1e-6;
const nt = 256; // lfm2_vision.cu's NT
const row_counts = [_]usize{ 1, 7, 308, 1024 }; // 308: a thumbnail's 22 x 14 patches; 1024: a whole tile

// Features against vision.json (torch bf16, CPU), relative L2: twice a CPU emulation of these roundings, reordered.
const norm_tolerance = 0.06; // every row's norm
const sum_tolerance = 0.17; // every row's sum
const row_tolerance = 0.035; // the picked rows whole
const cos_least = 0.998; // each picked row's cosine similarity
const gelu_share = 0.15; // GELU elements off: 2-ulp tanhf and erff flip bf16 roundings in the cancelling negative tail
// Attention against f64 SDPA: bf16 output rounding (2^-8 relative), plus f32 score, softmax and value-sum error.
const attention_tolerance = 0x1p-8; // times |want|
const attention_slack = 0x1p-10; // the f32 error's bound for |q|, |k|, |v| < 2, 72 dims and up to 1024 keys
const attention_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, dim))); // vision.attention's

extern fn erf(x: f64) f64;

/// warp_sum: xor butterflies over 32 lanes, lane 0's sum.
fn warpSum(v: *[32]f32) f32 {
    var off: usize = 16;
    while (off > 0) : (off >>= 1) {
        var n: [32]f32 = undefined;
        for (&n, 0..) |*x, l| x.* = v[l] + v[l ^ off];
        v.* = n;
    }
    return v[0];
}

/// block_sum over NT lanes: each warp's butterfly, then the warps in order.
fn blockSum(lanes: *[nt]f32) f32 {
    var s: f32 = 0;
    for (0..nt / 32) |w| {
        const ws = warpSum(lanes[w * 32 ..][0..32]);
        s = if (w == 0) ws else s + ws;
    }
    return s;
}

fn layerNorm(x: []const u16, w: []const u16, b: []const u16, y: []u16, rows: usize, width: usize) void {
    var vals: [8 * nt]f32 = undefined;
    for (0..rows) |r| {
        const at = r * width;
        var lanes: [nt]f32 = @splat(0);
        for (0..width) |i| {
            vals[i] = bf(x[at + i]);
            lanes[i % nt] += vals[i];
        }
        const mean = blockSum(&lanes) / @as(f32, @floatFromInt(width));
        lanes = @splat(0);
        for (0..width) |i| {
            vals[i] -= mean;
            lanes[i % nt] += vals[i] * vals[i];
        }
        const rstd = 1.0 / @sqrt(blockSum(&lanes) / @as(f32, @floatFromInt(width)) + eps);
        for (0..width) |i| y[at + i] = gm.f32ToBf16(vals[i] * rstd * bf(w[i]) + bf(b[i]));
    }
}

/// torch's tanh GELU in fp32 as lfm2_vision.cu runs it, tanh correctly rounded.
fn geluTanh(x: f32) f32 {
    const beta: f32 = @floatCast(std.math.sqrt2 * std.math.two_sqrtpi * 0.5);
    const inner = beta * (x + 0.044715 * (x * x * x));
    return 0.5 * x * (1 + @as(f32, @floatCast(std.math.tanh(@as(f64, inner)))));
}

/// torch's exact GELU in fp32 as lfm2_vision.cu runs it, erf correctly rounded.
fn geluErf(x: f32) f32 {
    const alpha: f32 = @floatCast(std.math.sqrt1_2);
    return x * 0.5 * (1 + @as(f32, @floatCast(erf(@as(f64, x * alpha)))));
}

fn bias(acc: []const f32, b: []const u16, r: ?[]const u16, o: []u16, n: usize, act: vision.Act) void {
    for (acc, o, 0..) |x, *out, i| {
        var v = gm.rbf(x + bf(b[i % n]));
        v = switch (act) {
            .none => v,
            .gelu_tanh => gm.rbf(geluTanh(v)),
            .gelu => gm.rbf(geluErf(v)),
        };
        out.* = gm.f32ToBf16(if (r) |res| bf(res[i]) + v else v);
    }
}

/// tf_lfm2v_attend: each row's f32 softmax (key j in lane j mod 32, then butterflies), an fma chain over the keys.
fn attend(scores: []const f32, qkv: []const u16, out: []u16, rows: usize, ss: usize, scale: f32) void {
    const w = 3 * heads * dim;
    var p: [1024]f32 = undefined;
    for (0..heads) |h| for (0..rows) |i| {
        const s = scores[h * ss + i * rows ..][0..rows];
        var m = -std.math.inf(f32);
        for (s, p[0..rows]) |x, *y| {
            y.* = x * scale;
            m = @max(m, y.*);
        }
        var lanes: [32]f32 = @splat(0);
        for (p[0..rows], 0..) |*y, j| {
            y.* = gm.exp(y.* - m);
            lanes[j % 32] += y.*;
        }
        const sum = warpSum(&lanes);
        for (p[0..rows]) |*y| y.* /= sum;
        for (0..dim) |d| {
            var o: f32 = 0;
            for (p[0..rows], 0..) |pj, j| o = gm.fma(pj, bf(qkv[j * w + 2 * heads * dim + h * dim + d]), o);
            out[i * heads * dim + h * dim + d] = gm.f32ToBf16(o);
        }
    };
}

/// SDPA in f64 over bf16 q, k, v rows of qkv [rows, 3 H D] -> [rows, H D].
fn attention(qkv: []const u16, out: []f64, rows: usize) void {
    const w = 3 * heads * dim;
    const scale = 1.0 / @sqrt(@as(f64, dim));
    var e: [1024]f64 = undefined;
    for (0..heads) |h| for (0..rows) |i| {
        var m = -std.math.inf(f64);
        for (0..rows) |j| {
            var dot: f64 = 0;
            for (0..dim) |d| dot += @as(f64, bf(qkv[i * w + h * dim + d])) * bf(qkv[j * w + heads * dim + h * dim + d]);
            e[j] = dot * scale;
            m = @max(m, e[j]);
        }
        var sum: f64 = 0;
        for (e[0..rows]) |*x| {
            x.* = @exp(x.* - m);
            sum += x.*;
        }
        for (0..dim) |d| {
            var acc: f64 = 0;
            for (0..rows) |j| acc += e[j] * bf(qkv[j * w + 2 * heads * dim + h * dim + d]);
            out[i * heads * dim + h * dim + d] = acc / sum;
        }
    };
}

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator, // host data, freed together at the end
    s: cuda.Stream,
    k: vision.Kernels,
    g: vision.Gemms,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bad: usize = 0,
    cases: usize = 0,

    /// A device copy of host values, landed before any kernel on the rig's stream reads it.
    fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        try r.gpu.ctx.synchronize();
        return b.ptr;
    }

    fn zeros(r: *Rig, comptime T: type, n: usize) !u64 {
        const host = try r.a.alloc(T, n);
        @memset(std.mem.sliceAsBytes(host), 0);
        return r.dev(T, host);
    }

    fn bfs(r: *Rig, n: usize, scale: f32) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| v.* = gm.f32ToBf16((r.rng.float(f32) * 2 - 1) * scale);
        return out;
    }

    fn f32s(r: *Rig, n: usize, scale: f32) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = (r.rng.float(f32) * 2 - 1) * scale;
        return out;
    }

    fn get(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        try r.s.synchronize();
        const out = try r.a.alloc(T, n);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, n * @sizeOf(T)), "cuMemcpyDtoH");
        return out;
    }

    /// bf16 rows at `ptr` against `want`: equal (share 0), else `share` of them off by a bf16 step or 2^-20; NaN fails.
    fn near(r: *Rig, what: []const u8, rows: usize, ptr: u64, want: []const u16, share: f64) !void {
        const got = try r.get(u16, ptr, want.len);
        r.cases += 1;
        var worst: f64 = 0;
        var differ: usize = 0;
        var far: usize = 0;
        var live: usize = 0;
        for (got, want) |g, w| {
            const err = @abs(@as(f64, bf(g)) - bf(w));
            worst = @max(worst, err);
            differ += @intFromBool(g != w);
            far += @intFromBool(!(err <= 0x1p-7 * @abs(bf(w)) + 0x1p-20));
            live += @intFromBool(bf(w) != 0);
        }
        std.debug.print("RESULT {s} rows {d}: max abs error {e}, {d} of {d} elements differ, {d} past a bf16 step\n", .{ what, rows, worst, differ, want.len, far });
        if (@as(f64, @floatFromInt(differ)) > share * @as(f64, @floatFromInt(want.len)) or far != 0 or live == 0) r.bad += 1;
    }
};

fn normCases(r: *Rig, rows: usize) !void {
    const x = try r.bfs(rows * hidden, 3.0);
    const w = try r.bfs(hidden, 1.0);
    const b = try r.bfs(hidden, 0.5);
    const yd = try r.zeros(u16, rows * hidden);
    try r.k.layerNorm(r.s, try r.dev(u16, x), try r.dev(u16, w), try r.dev(u16, b), yd, rows, hidden, eps);
    const y = try r.a.alloc(u16, rows * hidden);
    layerNorm(x, w, b, y, rows, hidden);
    try r.near("layer_norm", rows, yd, y, 0);
}

fn biasCases(r: *Rig, rows: usize) !void {
    for ([_]vision.Act{ .none, .gelu_tanh, .gelu }, [_]usize{ hidden, ffn, 2048 }) |act, n| for ([_]bool{ false, true }) |with_r| {
        if (with_r and act != .none) continue;
        const acc = try r.f32s(rows * n, 4.0);
        const b = try r.bfs(n, 1.0);
        const res = try r.bfs(rows * n, 2.0);
        const od = if (with_r) try r.dev(u16, res) else try r.zeros(u16, rows * n);
        try r.k.bias(r.s, try r.dev(f32, acc), try r.dev(u16, b), if (with_r) od else null, od, rows, n, act);
        const o = try r.a.alloc(u16, rows * n);
        bias(acc, b, if (with_r) res else null, o, n, act);
        try r.near(if (with_r) "bias + residual in place" else @tagName(act), rows, od, o, if (act == .none) 0 else gelu_share);
    };
}

/// The head split byte for byte, tf_lfm2v_attend against its mirror, then the whole attention against f64.
fn attentionCases(r: *Rig, rows: usize) !void {
    const st = vision.Strides.of(rows, dim);
    const qkv = try r.bfs(rows * 3 * heads * dim, 2.0);
    const qkvd = try r.dev(u16, qkv);
    const hb: vision.Heads = .{ .q = try r.zeros(u16, heads * st.hs), .k = try r.zeros(u16, heads * st.hs), .scores = try r.zeros(f32, heads * st.ss) };
    try r.k.splitHeads(r.s, qkvd, hb.q, hb.k, rows, heads, dim, st.hs);
    const q = try r.a.alloc(u16, heads * st.hs);
    const k = try r.a.alloc(u16, heads * st.hs);
    for ([_][]u16{ q, k }) |m| @memset(m, 0);
    for (0..rows) |i| for (0..heads) |h| for (0..dim) |d| {
        const at = i * 3 * heads * dim + h * dim + d;
        q[h * st.hs + i * dim + d] = qkv[at];
        k[h * st.hs + i * dim + d] = qkv[at + heads * dim];
    };
    try r.near("split_heads q", rows, hb.q, q, 0);
    try r.near("split_heads k", rows, hb.k, k, 0);
    const scores = try r.f32s(heads * st.ss, 30.0);
    const ad = try r.zeros(u16, rows * heads * dim);
    try r.k.attend(r.s, try r.dev(f32, scores), qkvd, ad, rows, heads, dim, st.ss, 0.125);
    const mirrored = try r.a.alloc(u16, rows * heads * dim);
    attend(scores, qkv, mirrored, rows, st.ss, 0.125);
    try r.near("attend", rows, ad, mirrored, 0);
    try vision.attention(&r.k, &r.g, r.s, hb, qkvd, ad, rows, heads, dim);
    // QK^T: exact bf16 products summed in f32 in any order, so each score is within (D + 1) 2^-24 sum |q k| of the dot
    const own_scores = try r.get(f32, hb.scores, heads * st.ss);
    var score_far: usize = 0;
    for (0..heads) |h| for (0..rows) |i| for (0..rows) |j| {
        var dot: f64 = 0;
        var mag: f64 = 0;
        for (0..dim) |d| {
            const x = @as(f64, bf(q[h * st.hs + i * dim + d])) * bf(k[h * st.hs + j * dim + d]);
            dot += x;
            mag += @abs(x);
        }
        score_far += @intFromBool(!(@abs(own_scores[h * st.ss + i * rows + j] - dot) <= (dim + 1) * 0x1p-24 * mag));
    };
    r.cases += 1;
    std.debug.print("RESULT attention scores rows {d}: {d} of {d} past (D + 1) 2^-24 sum |q k|\n", .{ rows, score_far, heads * rows * rows });
    if (score_far != 0) r.bad += 1;
    const own = try r.a.alloc(u16, rows * heads * dim);
    attend(own_scores, qkv, own, rows, st.ss, attention_scale);
    try r.near("attention on its own scores", rows, ad, own, 0);
    const got = try r.get(u16, ad, rows * heads * dim);
    const want = try r.a.alloc(f64, rows * heads * dim);
    attention(qkv, want, rows);
    var worst: f64 = 0;
    var far: usize = 0;
    for (got, want) |g, w| {
        const err = @abs(bf(g) - w);
        worst = @max(worst, err);
        far += @intFromBool(!(err <= attention_tolerance * @abs(w) + attention_slack));
    }
    r.cases += 1;
    std.debug.print("RESULT attention rows {d}: max abs error {e} against f64, {d} past 2^-8 |want| + 2^-10\n", .{ rows, worst, far });
    if (far != 0) r.bad += 1;
}

fn unshuffleCase(r: *Rig, h: usize, w: usize) !void {
    const x = try r.bfs(h * w * hidden, 1.0);
    const yd = try r.zeros(u16, h * w * hidden);
    try r.k.unshuffle(r.s, try r.dev(u16, x), yd, h, w, hidden);
    const y = try r.a.alloc(u16, h * w * hidden);
    for (0..h / 2) |i| for (0..w / 2) |j| for (0..4) |q| {
        const src = ((2 * i + q / 2) * w + 2 * j + q % 2) * hidden;
        @memcpy(y[((i * (w / 2) + j) * 4 + q) * hidden ..][0..hidden], x[src..][0..hidden]);
    };
    try r.near("unshuffle", h * w, yd, y, 0);
}

fn json(a: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Value {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    return std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
}

fn number(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

/// Each golden picture through image.zig and the tower: norms and sums of every row, the picked rows whole.
fn features(r: *Rig, model: []const u8) !void {
    const gpa = r.gpu.gpa;
    const io = r.gpu.io;
    var problem: []const u8 = "";
    var cfg = try lfm2.config.Config.load(gpa, io, model, &problem) orelse {
        std.debug.print("FAIL lfm2-vision: {s}\n", .{problem});
        return error.TestFailed;
    };
    defer cfg.deinit(gpa);
    var w = try lfm2.weights.Weights.load(gpa, io, r.gpu.d, model, &cfg);
    defer w.deinit(gpa);
    var v = try vision.Vision.init(gpa, r.gpu.d, &cfg, &(w.vision orelse return error.NoVisionTower));
    defer v.deinit();
    const urls = (try json(r.a, io, "zig/tests/fixtures/lfm2/images.json")).object.get("cases").?.array.items;
    const golden = (try json(r.a, io, "zig/tests/fixtures/lfm2/vision.json")).object.get("cases").?.array.items;
    for (golden) |case| {
        const name = case.object.get("name").?.string;
        const url = for (urls) |u| {
            if (std.mem.eql(u8, u.object.get("name").?.string, name)) break u.object.get("url").?.string;
        } else return error.MissingField;
        var rgb = (try lfm2.image.load(r.a, url, &problem)) orelse return error.TestFailed;
        defer rgb.deinit(r.a);
        var pic = try lfm2.image.prepare(r.a, rgb);
        defer pic.deinit(r.a);
        const rows: usize = @intCast(case.object.get("rows").?.integer);
        const width = cfg.hidden;
        const outd = try r.zeros(u16, rows * width);
        const got_rows = try v.features(r.s, &pic, outd);
        try check.expect(got_rows == rows, "lfm2-vision {s}: {d} feature rows, the golden has {d}", .{ name, got_rows, rows });
        const got = try r.get(u16, outd, rows * width);
        var nonfinite: usize = 0;
        for (got) |g| nonfinite += @intFromBool(!std.math.isFinite(bf(g)));
        var norm_err = [2]f64{ 0, 0 }; // squared differences, squared golden values
        var sum_err = [2]f64{ 0, 0 };
        for (case.object.get("norms").?.array.items, case.object.get("sums").?.array.items, 0..) |nv, sv, i| {
            var norm: f64 = 0;
            var sum: f64 = 0;
            for (got[i * width ..][0..width]) |g| {
                norm += @as(f64, bf(g)) * bf(g);
                sum += bf(g);
            }
            norm_err[0] += std.math.pow(f64, @sqrt(norm) - number(nv), 2);
            norm_err[1] += number(nv) * number(nv);
            sum_err[0] += std.math.pow(f64, sum - number(sv), 2);
            sum_err[1] += number(sv) * number(sv);
        }
        const encoded = case.object.get("values").?.string;
        const bytes = try r.a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(encoded));
        try std.base64.standard.Decoder.decode(bytes, encoded);
        var row_err = [2]f64{ 0, 0 };
        var cos_worst: f64 = 1;
        for (case.object.get("picked").?.array.items, 0..) |pv, j| {
            const row: usize = @intCast(pv.integer);
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            for (got[row * width ..][0..width], 0..) |g, e| {
                const want: f64 = bf(std.mem.readInt(u16, bytes[(j * width + e) * 2 ..][0..2], .little));
                dot += bf(g) * want;
                gg += @as(f64, bf(g)) * bf(g);
                ww += want * want;
                row_err[0] += std.math.pow(f64, bf(g) - want, 2);
            }
            row_err[1] += ww;
            const cos = dot / @sqrt(gg * ww);
            if (!(cos >= cos_worst)) cos_worst = cos;
        }
        const norms = @sqrt(norm_err[0] / norm_err[1]);
        const sums = @sqrt(sum_err[0] / sum_err[1]);
        const picked = @sqrt(row_err[0] / row_err[1]);
        r.cases += 1;
        std.debug.print("RESULT features {s}: {d} tiles, {d} rows, {d} non-finite; relative L2 error of row norms {d:.4}, row sums {d:.4}, picked rows {d:.4}; least picked cosine {d:.5}\n", .{ name, pic.tiles.len, rows, nonfinite, norms, sums, picked, cos_worst });
        if (nonfinite != 0 or !(norms <= norm_tolerance and sums <= sum_tolerance and picked <= row_tolerance and cos_worst >= cos_least)) r.bad += 1;
    }
}

/// The vision kernels on seeded rows against host references; with MODEL, the goldens' features too.
pub fn run(gpu: Gpu, model: ?[]const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x5167_1192);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, true), .k = try vision.Kernels.load(gpu.d), .g = try vision.Gemms.init(gpu.gpa, gpu.d), .rng = prng.random() };
    defer {
        r.g.deinit();
        r.k.unload();
        r.s.deinit();
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    for (row_counts) |rows| {
        try normCases(&r, rows);
        try biasCases(&r, rows);
        try attentionCases(&r, rows);
    }
    try unshuffleCase(&r, 32, 32);
    try unshuffleCase(&r, 22, 14);
    if (model) |m| try features(&r, m) else std.debug.print("SKIP lfm2-vision features: no MODEL\n", .{});
    try check.expect(r.bad == 0, "lfm2-vision: {d} of {d} comparisons past their references", .{ r.bad, r.cases });
    check.pass("lfm2-vision: {d} comparisons within their references (kernels byte-equal, GELU within a bf16 step, attention and features within stated bounds)", .{r.cases});
}

//! d1 (LFM2-VL) on CUDA: an engine that scores decisions through the score hook and writes no text.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lfm2 = @import("lfm2");
const Allocator = std.mem.Allocator;

pub const family: api.Family = .{ .model_type = "lfm2_vl", .formats = &.{"bf16"} };
const default_context = 32768;

const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    cfg: lfm2.config.Config,
    weights: lfm2.weights.Weights,
    model: lfm2.model.Model,
    window: u32,
    mutex: std.Io.Mutex = .init, // one prompt pass at a time: HTTP threads call score directly

    fn of(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .name = "lfm2", .context_window = of(ctx).window, .thinking_budget = false, .decides = true };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed; // the server refuses text requests before they reach a decision engine
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak };
    }

    fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
        const e = of(ctx);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        e.ctx.makeCurrent() catch return error.Failed;
        return e.model.score(prompt, labels, logits, mass) catch |err| {
            std.log.err("lfm2 decision pass failed: {s}", .{@errorName(err)});
            return error.Failed;
        };
    }

    fn close(ctx: *anyopaque) void {
        const e = of(ctx);
        e.ctx.makeCurrent() catch {};
        e.model.deinit();
        e.weights.deinit(e.gpa);
        e.cfg.deinit(e.gpa);
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .score = score };
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const window: u32 = if (o.context) |c| blk: {
        if (c < 1 or c > default_context) {
            problem.* = try std.fmt.allocPrint(a, "--context {d}: d1 reads 1 to {d} tokens a decision", .{ c, default_context });
            return null;
        }
        break :blk @intCast(c);
    } else default_context;
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .window = window, .driver = undefined, .ctx = undefined, .cfg = undefined, .weights = undefined, .model = undefined };
    e.cfg = try lfm2.config.Config.load(gpa, io, o.dir, problem) orelse return null;
    errdefer e.cfg.deinit(gpa);
    e.driver = cuda.Driver.open() catch |err| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(err)});
        return null;
    };
    errdefer e.driver.close();
    e.ctx = try cuda.Context.init(&e.driver, 0);
    errdefer e.ctx.deinit();
    try e.ctx.makeCurrent();
    e.weights = try lfm2.weights.Weights.load(gpa, io, &e.driver, o.dir, &e.cfg);
    errdefer e.weights.deinit(gpa);
    e.model = try lfm2.model.Model.init(gpa, &e.driver, &e.cfg, &e.weights, window);
    return .{ .engine = .{ .ctx = e, .vtable = &Engine.vtable }, .close = Engine.close, .ctx = e };
}

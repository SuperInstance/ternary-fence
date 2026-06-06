# ternary-fence

Synchronization fences for coordinated ternary computation.

When you're running distributed ternary inference — forward passes, backward passes, gradient aggregation — you need threads to agree on ordering. Not with heavyweight channels or async runtimes, but with lightweight one-shot barriers that cost almost nothing to create and reuse. That's what this crate provides.

The mental model is borrowed from GPU fence semantics (Vulkan fences, CUDA events): signal once, wait many, reset and recycle. But everything here runs on the CPU with standard library primitives — no GPU required.

## Why This Exists

Ternary neural networks split work across threads and pipeline stages. The data flow looks like:

```
Producer thread → [fence] → Consumer thread
Worker 1 → [fence] → Gradient aggregator → [fence] → Parameter updater
```

Each arrow is a fence. You need them to be:
- **Cheap** — you'll create thousands per training step
- **Reusable** — don't allocate new ones every iteration
- **Timeout-capable** — detect stalled workers without blocking forever
- **Observable** — know which stage is blocking progress

`std::sync::Barrier` almost works, but it's N-thread oriented and doesn't timeout. `std::sync::mpsc` adds allocation overhead. Fences hit the sweet spot: signal/wait semantics with pool-based reuse.

## Quick Start

```rust
use ternary_fence::{Fence, FencePool, FenceManager};
use std::thread;
use std::sync::Arc;

// Basic: signal and wait across threads
let fence = Arc::new(Fence::with_label("forward_pass_done"));

let f = fence.clone();
let h = thread::spawn(move || {
    // ... do expensive computation ...
    f.signal();
});

fence.wait(); // blocks until signaled
h.join().unwrap();
```

With pooling for repeated use:

```rust
let pool = Arc::new(FencePool::new());

// Each training iteration
let fence = pool.acquire();           // reuses from pool or creates new
fence.signal();
pool.release(fence);                  // returns for reuse

println!("Pool stats: created={}, reused={}", 
    pool.created_count(), pool.reused_count());
```

## Architecture

The crate has four layers, each building on the previous:

```
┌─────────────────────────────────────┐
│         FenceManager                │  Lifecycle tracking + pool integration
│  (create_fence, signal_fence,       │
│   wait_all, cleanup_completed)      │
├─────────────────────────────────────┤
│         FencePool                   │  Object reuse to avoid allocation
│  (acquire, release, stats)          │
├─────────────────────────────────────┤
│         ProducerConsumer            │  Ordered data flow pattern
│  (produce, consume_all, fences)     │
├─────────────────────────────────────┤
│         Fence                       │  Core primitive: signal/wait
│  (signal, wait, wait_timeout,       │
│   reset, is_signaled, clone)        │
└─────────────────────────────────────┘
```

### Fence (core primitive)

A one-time barrier backed by `Mutex<FenceState> + Condvar`. Signal it, wait on it, reset it for reuse. Cloning shares the underlying state via `Arc`, so all clones see the same signal.

```rust
let fence = Fence::new();
let clone = fence.clone();  // cheap Arc clone

fence.signal();
assert!(clone.is_signaled()); // clone sees the signal too
```

The state machine is simple:

```
Unsignaled ──signal()──→ Signaled(Instant)
    ↑                        │
    └────reset()─────────────┘
```

Once signaled, all past and future `wait()` calls return immediately. Reset brings it back to blocking.

### FencePool (reuse layer)

Each `Fence` allocates a `Mutex + Condvar + Arc`. In a training loop creating thousands of fences per iteration, that adds up. `FencePool` recycles them:

```rust
let pool = FencePool::new();

// Iteration 1: creates new fence
let f1 = pool.acquire();  // created_count = 1, reused_count = 0
pool.release(f1);

// Iteration 2: reuses the same fence (reset automatically)
let f2 = pool.acquire();  // created_count = 1, reused_count = 1
```

The pool tracks statistics so you can verify reuse rates in production.

### ProducerConsumer (pattern layer)

A ready-made pattern for the most common fence use case: ordered data production and consumption.

```rust
use ternary_fence::{Fence, ProducerConsumer};

let mut pc = ProducerConsumer::new();
let done = Fence::new();
pc.set_producer_fence(done.clone());

// Producer side
pc.produce("tensor_batch_1");
pc.produce("tensor_batch_2");
pc.signal_producer_done();

// Consumer side
pc.wait_producer();              // blocks until producer signals
let items = pc.consume_all();    // ["tensor_batch_1", "tensor_batch_2"]
```

### FenceManager (lifecycle layer)

When you have dozens of active fences — one per pipeline stage, one per worker — tracking their state manually is error-prone. `FenceManager` handles it:

```rust
let manager = FenceManager::new();

let f1 = manager.create_labeled_fence("forward_pass");
let f2 = manager.create_labeled_fence("backward_pass");

// In worker threads:
manager.signal_fence(f1.id());

// In coordinator:
manager.wait_all();  // blocks until every active fence is signaled
manager.cleanup_completed();  // returns completed fences to pool
```

Status tracking: `Active` → `Completed` or `TimedOut`. Query with `get_status()`, `active_count()`, `completed_count()`.

## API Reference

### `Fence`

| Method | Description |
|--------|-------------|
| `new()` | Create unsignaled fence |
| `with_label(label)` | Create with debug label |
| `signal()` | Signal, unblocking all waiters |
| `wait()` | Block until signaled |
| `wait_timeout(duration)` | Block with timeout, returns `Result` |
| `is_signaled()` | Non-blocking check |
| `reset()` | Return to unsignaled state |
| `signaled_at()` | When the fence was signaled |
| `clone()` | Share state via Arc |

### `FencePool`

| Method | Description |
|--------|-------------|
| `new()` | Empty pool |
| `acquire()` | Get a fence (new or reused) |
| `acquire_labeled(label)` | Get a labeled fence |
| `release(fence)` | Return fence to pool |
| `available()` | Fences currently in pool |
| `created_count()` / `reused_count()` | Allocation statistics |

### `FenceManager`

| Method | Description |
|--------|-------------|
| `new()` / `with_pool(pool)` | Create manager |
| `create_fence()` / `create_labeled_fence(label)` | Register new active fence |
| `signal_fence(id)` | Signal and mark completed |
| `wait_fence(id)` / `wait_fence_timeout(id, dur)` | Wait for specific fence |
| `wait_all()` | Wait for every active fence |
| `get_status(id)` | Query fence lifecycle state |
| `cleanup_completed()` | Return completed fences to pool |

### `ProducerConsumer`

| Method | Description |
|--------|-------------|
| `produce(item)` | Add item to buffer |
| `consume_all()` | Take all buffered items |
| `signal_producer_done()` / `signal_consumer_done()` | Signal fence |
| `wait_producer()` / `wait_consumer()` | Wait on fence |

## Real-World Example: Pipeline Stage Coordination

Here's how you'd wire up a three-stage ternary inference pipeline:

```rust
use ternary_fence::{FenceManager, FencePool, FenceStatus};
use std::sync::Arc;
use std::thread;

let pool = Arc::new(FencePool::new());
let manager = Arc::new(FenceManager::with_pool(pool));

// Stage 1: data loading
let mgr1 = manager.clone();
let t1 = thread::spawn(move || {
    let fence = mgr1.create_labeled_fence("data_loaded");
    // ... load ternary weights from disk ...
    mgr1.signal_fence(fence.id());
});

// Stage 2: forward pass (waits for data)
let mgr2 = manager.clone();
let t2 = thread::spawn(move || {
    let data_fence = mgr2.create_labeled_fence("data_loaded");
    mgr2.wait_fence(data_fence.id());  // wait for stage 1
    
    let fwd_fence = mgr2.create_labeled_fence("forward_done");
    // ... run forward pass ...
    mgr2.signal_fence(fwd_fence.id());
});

t1.join().unwrap();
t2.join().unwrap();

manager.cleanup_completed();  // return all fences to pool
```

## Ecosystem Connections

This crate is the coordination layer for the ternary compute stack:

- **`ternary-warp-block`** — warp-level primitives that may signal fences on completion
- **`ternary-grid-launch`** — kernel launch configs that determine how many fences you need
- **`ternary-shared-memory`** — shared memory operations that fences synchronize
- **`ternary-em`** / **`ternary-logistic`** / **`ternary-regression`** — training algorithms that use fences for gradient sync

## Performance Notes

- **Fence creation**: ~100ns (Mutex + Condvar + Arc allocation). Pool reuse eliminates this.
- **Signal**: O(1) — lock, set state, notify_all on condvar.
- **Wait (already signaled)**: O(1) — checks state and returns immediately.
- **Wait (blocking)**: OS-level futex/condvar wait. Zero CPU usage while blocked.
- **Clone**: One Arc increment. Share fences freely between threads.
- **Pool acquire**: One Mutex lock + Vec pop. Faster than `Fence::new()` by ~10x.

Thread safety: all types are `Send + Sync`. The `Mutex + Condvar` pair handles synchronization internally.

## Open Questions

- **Async support**: Currently blocking only. An async `wait()` would require a different internal mechanism (tokio `Notify` or similar). Contributions welcome.
- **Fence chaining**: No built-in "signal fence B when fence A fires." You can build it with a thread, but a native `Fence::then()` would be cleaner.
- **Spurious wakeup handling**: The `wait()` loop rechecks state after wakeup, so spurious wakeups are handled correctly by design.

## License

MIT

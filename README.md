# ternary-fence

**Synchronization fences** for distributed ternary computation. Provides signal/wait fences, fence pooling for object reuse, fence lifecycle management, timeout handling, and producer-consumer ordering patterns — all in pure Rust.

## Why?

Distributed ternary inference needs coordination primitives:

- **Pipeline stages** — Signal completion between forward/backward passes
- **Data parallelism** — Sync workers at gradient aggregation barriers
- **Resource management** — Reuse fence objects to avoid allocation overhead
- **Timeout safety** — Detect stalled computations without blocking forever
- **Producer-consumer** — Order data production and consumption across threads

This crate gives you lightweight, reusable fences inspired by GPU fence semantics (Vulkan/CUDA), adapted for CPU-side ternary workloads.

## Quick Start

```rust
use ternary_fence::{Fence, FencePool, FenceManager};

// Basic signal + wait
let fence = Fence::new();
fence.signal();
fence.wait(); // returns immediately — already signaled
assert!(fence.is_signaled());
```

## Core Concepts

### Fence

A one-time synchronization primitive. Signal it, wait on it, reset for reuse:

```rust
let fence = Fence::with_label("layer_3_done");

// In producer thread:
fence.signal();

// In consumer thread:
fence.wait(); // blocks until signaled
```

### FencePool

Avoid allocation overhead by pooling fence objects:

```rust
let pool = FencePool::new();

let f1 = pool.acquire();      // creates new
pool.release(f1);             // returns to pool

let f2 = pool.acquire();      // reuses from pool (same fence, reset)
println!("Created: {}, Reused: {}", pool.created_count(), pool.reused_count());
```

### FenceManager

Track multiple active fences with lifecycle management:

```rust
let manager = FenceManager::new();

let f1 = manager.create_labeled_fence("forward_pass");
let f2 = manager.create_labeled_fence("backward_pass");

// Signal and track completion
manager.signal_fence(f1.id());
assert_eq!(manager.active_count(), 1);
assert_eq!(manager.completed_count(), 1);

// Cleanup: return completed fences to pool
let cleaned = manager.cleanup_completed();
```

### Timeouts

Don't block forever — detect stalled computations:

```rust
let fence = Fence::new();
match fence.wait_timeout(std::time::Duration::from_secs(5)) {
    Ok(()) => println!("Completed!"),
    Err(()) => println!("Timed out — something is stuck"),
}
```

### Producer-Consumer

Built-in pattern for ordered data flow:

```rust
use ternary_fence::{Fence, ProducerConsumer};

let mut pc = ProducerConsumer::new();
let done = Fence::new();
pc.set_producer_fence(done.clone());

pc.produce("tensor_batch_1");
pc.produce("tensor_batch_2");
pc.signal_producer_done(); // signals the fence

// Consumer side:
pc.wait_producer();  // waits for production to finish
let items = pc.consume_all(); // ["tensor_batch_1", "tensor_batch_2"]
```

## API Surface

| Type | Description |
|------|-------------|
| `Fence` | Signal/wait synchronization primitive |
| `FencePool` | Reusable fence object pool |
| `FenceManager` | Track and manage multiple fences |
| `FenceStatus` | Active / Completed / TimedOut enum |
| `ProducerConsumer` | Ordered produce/consume with fence coordination |
| `FenceId` | Unique fence identifier |

## Features

- **Thread-safe** — All types are `Send + Sync`, safe for multi-threaded use
- **Zero-allocation reuse** — FencePool returns reset fences without new allocations
- **Timeout support** — Every wait has a timeout variant
- **Clonable** — Fences clone cheaply (shared state via `Arc`)
- **Labeled** — Optional labels for debugging and logging

## Testing

```bash
cargo test
```

24 tests covering signal/wait, cross-thread synchronization, pool reuse, timeout detection, producer-consumer ordering, fence management lifecycle, multiple waiters, and fence cloning.

## License

MIT

//! # ternary-fence
//!
//! Synchronization fences for distributed ternary computation.
//! Provides signal/wait fences, fence pooling, fence management,
//! timeout handling, and producer-consumer ordering patterns.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Unique identifier for a fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FenceId(pub u64);

static NEXT_FENCE_ID: AtomicU64 = AtomicU64::new(1);

/// Internal state of a fence.
#[derive(Debug)]
enum FenceState {
    /// Fence has been signaled with a timestamp.
    Signaled(Instant),
    /// Fence is unsignaled.
    Unsignaled,
}

/// A synchronization fence that supports signal and wait operations.
///
/// Fences act as one-time barriers. Once signaled, all current and future
/// wait calls return immediately. Fences can be reset for reuse.
#[derive(Debug)]
pub struct Fence {
    id: FenceId,
    state: Arc<(Mutex<FenceState>, Condvar)>,
    label: String,
}

impl Fence {
    /// Create a new unsignaled fence.
    pub fn new() -> Self {
        Fence {
            id: FenceId(NEXT_FENCE_ID.fetch_add(1, Ordering::SeqCst)),
            state: Arc::new((Mutex::new(FenceState::Unsignaled), Condvar::new())),
            label: String::new(),
        }
    }

    /// Create a new labeled fence.
    pub fn with_label(label: impl Into<String>) -> Self {
        Fence {
            id: FenceId(NEXT_FENCE_ID.fetch_add(1, Ordering::SeqCst)),
            state: Arc::new((Mutex::new(FenceState::Unsignaled), Condvar::new())),
            label: label.into(),
        }
    }

    /// Get the fence's unique identifier.
    pub fn id(&self) -> FenceId {
        self.id
    }

    /// Get the fence's label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Signal this fence, unblocking all waiters.
    pub fn signal(&self) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        *state = FenceState::Signaled(Instant::now());
        cvar.notify_all();
    }

    /// Wait until this fence is signaled. Blocks indefinitely.
    pub fn wait(&self) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        while matches!(*state, FenceState::Unsignaled) {
            state = cvar.wait(state).unwrap();
        }
    }

    /// Wait with a timeout. Returns `Ok(())` if signaled, `Err(())` on timeout.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<(), ()> {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        let deadline = Instant::now() + timeout;
        loop {
            if !matches!(*state, FenceState::Unsignaled) {
                return Ok(());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(());
            }
            let result = cvar.wait_timeout(state, remaining).unwrap();
            state = result.0;
            if result.1.timed_out() {
                if matches!(*state, FenceState::Unsignaled) {
                    return Err(());
                }
                return Ok(());
            }
        }
    }

    /// Check if the fence has been signaled without blocking.
    pub fn is_signaled(&self) -> bool {
        let state = self.state.0.lock().unwrap();
        matches!(*state, FenceState::Signaled(_))
    }

    /// Reset the fence to unsignaled state, allowing reuse.
    pub fn reset(&self) {
        let (lock, _) = &*self.state;
        let mut state = lock.lock().unwrap();
        *state = FenceState::Unsignaled;
    }

    /// Get the instant when the fence was signaled, if it has been.
    pub fn signaled_at(&self) -> Option<Instant> {
        let state = self.state.0.lock().unwrap();
        match &*state {
            FenceState::Signaled(t) => Some(*t),
            FenceState::Unsignaled => None,
        }
    }
}

impl Clone for Fence {
    fn clone(&self) -> Self {
        Fence {
            id: self.id,
            state: Arc::clone(&self.state),
            label: self.label.clone(),
        }
    }
}

impl Default for Fence {
    fn default() -> Self {
        Self::new()
    }
}

/// A pool of reusable fence objects. Fences are returned to the pool after use.
pub struct FencePool {
    pool: Arc<Mutex<Vec<Fence>>>,
    created_count: Arc<AtomicU64>,
    reused_count: Arc<AtomicU64>,
}

impl FencePool {
    /// Create a new empty fence pool.
    pub fn new() -> Self {
        FencePool {
            pool: Arc::new(Mutex::new(Vec::new())),
            created_count: Arc::new(AtomicU64::new(0)),
            reused_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Acquire a fence from the pool. Creates a new one if the pool is empty.
    pub fn acquire(&self) -> Fence {
        let mut pool = self.pool.lock().unwrap();
        if let Some(fence) = pool.pop() {
            fence.reset();
            self.reused_count.fetch_add(1, Ordering::SeqCst);
            fence
        } else {
            self.created_count.fetch_add(1, Ordering::SeqCst);
            Fence::new()
        }
    }

    /// Acquire a labeled fence from the pool.
    pub fn acquire_labeled(&self, label: impl Into<String>) -> Fence {
        let mut pool = self.pool.lock().unwrap();
        if let Some(mut fence) = pool.pop() {
            fence.reset();
            fence.label = label.into();
            self.reused_count.fetch_add(1, Ordering::SeqCst);
            fence
        } else {
            self.created_count.fetch_add(1, Ordering::SeqCst);
            Fence::with_label(label)
        }
    }

    /// Return a fence to the pool for reuse.
    pub fn release(&self, fence: Fence) {
        self.pool.lock().unwrap().push(fence);
    }

    /// Get the number of fences currently available in the pool.
    pub fn available(&self) -> usize {
        self.pool.lock().unwrap().len()
    }

    /// Get total number of fences created (not reused from pool).
    pub fn created_count(&self) -> u64 {
        self.created_count.load(Ordering::SeqCst)
    }

    /// Get total number of fences reused from pool.
    pub fn reused_count(&self) -> u64 {
        self.reused_count.load(Ordering::SeqCst)
    }

    /// Get total number of acquisitions (created + reused).
    pub fn total_acquisitions(&self) -> u64 {
        self.created_count() + self.reused_count()
    }

    /// Clear the pool.
    pub fn clear(&self) {
        self.pool.lock().unwrap().clear();
    }
}

impl Default for FencePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Status of a tracked fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceStatus {
    Active,
    Completed,
    TimedOut,
}

/// Manages multiple fences and tracks their lifecycle.
pub struct FenceManager {
    fences: Arc<Mutex<HashMap<FenceId, (Fence, FenceStatus)>>>,
    pool: Arc<FencePool>,
}

impl FenceManager {
    /// Create a new fence manager.
    pub fn new() -> Self {
        FenceManager {
            fences: Arc::new(Mutex::new(HashMap::new())),
            pool: Arc::new(FencePool::new()),
        }
    }

    /// Create a fence manager with a shared pool.
    pub fn with_pool(pool: Arc<FencePool>) -> Self {
        FenceManager {
            fences: Arc::new(Mutex::new(HashMap::new())),
            pool,
        }
    }

    /// Create and register a new active fence.
    pub fn create_fence(&self) -> Fence {
        let fence = self.pool.acquire();
        let id = fence.id();
        self.fences
            .lock()
            .unwrap()
            .insert(id, (fence.clone(), FenceStatus::Active));
        fence
    }

    /// Create and register a labeled fence.
    pub fn create_labeled_fence(&self, label: impl Into<String>) -> Fence {
        let fence = self.pool.acquire_labeled(label);
        let id = fence.id();
        self.fences
            .lock()
            .unwrap()
            .insert(id, (fence.clone(), FenceStatus::Active));
        fence
    }

    /// Signal a fence and mark it as completed.
    pub fn signal_fence(&self, id: FenceId) -> bool {
        let mut fences = self.fences.lock().unwrap();
        if let Some((fence, status)) = fences.get_mut(&id) {
            fence.signal();
            *status = FenceStatus::Completed;
            true
        } else {
            false
        }
    }

    /// Wait for a specific fence. Returns false if fence not found.
    pub fn wait_fence(&self, id: FenceId) -> bool {
        let fences = self.fences.lock().unwrap();
        if let Some((fence, _)) = fences.get(&id) {
            let fence = fence.clone();
            drop(fences);
            fence.wait();
            true
        } else {
            false
        }
    }

    /// Wait for a fence with timeout. Returns its new status.
    pub fn wait_fence_timeout(&self, id: FenceId, timeout: Duration) -> FenceStatus {
        let fences = self.fences.lock().unwrap();
        if let Some((fence, _)) = fences.get(&id) {
            let fence = fence.clone();
            drop(fences);
            match fence.wait_timeout(timeout) {
                Ok(()) => {
                    self.fences
                        .lock()
                        .unwrap()
                        .entry(id)
                        .and_modify(|(_, s)| *s = FenceStatus::Completed);
                    FenceStatus::Completed
                }
                Err(()) => {
                    self.fences
                        .lock()
                        .unwrap()
                        .entry(id)
                        .and_modify(|(_, s)| *s = FenceStatus::TimedOut);
                    FenceStatus::TimedOut
                }
            }
        } else {
            FenceStatus::TimedOut
        }
    }

    /// Get the status of a fence.
    pub fn get_status(&self, id: FenceId) -> Option<FenceStatus> {
        self.fences
            .lock()
            .unwrap()
            .get(&id)
            .map(|(_, status)| *status)
    }

    /// Get the number of active (uncompleted) fences.
    pub fn active_count(&self) -> usize {
        self.fences
            .lock()
            .unwrap()
            .values()
            .filter(|(_, s)| *s == FenceStatus::Active)
            .count()
    }

    /// Get the number of completed fences.
    pub fn completed_count(&self) -> usize {
        self.fences
            .lock()
            .unwrap()
            .values()
            .filter(|(_, s)| *s == FenceStatus::Completed)
            .count()
    }

    /// Get total number of tracked fences.
    pub fn total_count(&self) -> usize {
        self.fences.lock().unwrap().len()
    }

    /// Remove completed fences and return them to the pool.
    pub fn cleanup_completed(&self) -> usize {
        let mut fences = self.fences.lock().unwrap();
        let completed: Vec<FenceId> = fences
            .iter()
            .filter(|(_, (_, s))| *s == FenceStatus::Completed)
            .map(|(id, _)| *id)
            .collect();

        let count = completed.len();
        for id in completed {
            if let Some((fence, _)) = fences.remove(&id) {
                self.pool.release(fence);
            }
        }
        count
    }

    /// Wait for all active fences to be signaled.
    pub fn wait_all(&self) {
        let fences = self.fences.lock().unwrap();
        let active: Vec<(FenceId, Fence)> = fences
            .iter()
            .filter(|(_, (_, s))| *s == FenceStatus::Active)
            .map(|(id, (f, _))| (*id, f.clone()))
            .collect();
        drop(fences);

        for (id, fence) in active {
            fence.wait();
            self.fences
                .lock()
                .unwrap()
                .entry(id)
                .and_modify(|(_, s)| *s = FenceStatus::Completed);
        }
    }
}

impl Default for FenceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Producer-consumer pattern using fences for ordering.
pub struct ProducerConsumer {
    producer_fence: Option<Fence>,
    consumer_fence: Option<Fence>,
    items: Arc<Mutex<Vec<String>>>,
}

impl ProducerConsumer {
    /// Create a new producer-consumer pair.
    pub fn new() -> Self {
        ProducerConsumer {
            producer_fence: None,
            consumer_fence: None,
            items: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Produce an item and optionally signal the producer fence.
    pub fn produce(&mut self, item: impl Into<String>) {
        self.items.lock().unwrap().push(item.into());
    }

    /// Set the producer's fence. Consumers can wait on this.
    pub fn set_producer_fence(&mut self, fence: Fence) {
        self.producer_fence = Some(fence);
    }

    /// Set the consumer's fence. Producers can wait on this.
    pub fn set_consumer_fence(&mut self, fence: Fence) {
        self.consumer_fence = Some(fence);
    }

    /// Signal that production is complete.
    pub fn signal_producer_done(&self) {
        if let Some(ref fence) = self.producer_fence {
            fence.signal();
        }
    }

    /// Signal that consumption is complete.
    pub fn signal_consumer_done(&self) {
        if let Some(ref fence) = self.consumer_fence {
            fence.signal();
        }
    }

    /// Wait for the producer to finish.
    pub fn wait_producer(&self) {
        if let Some(ref fence) = self.producer_fence {
            fence.wait();
        }
    }

    /// Wait for the consumer to finish.
    pub fn wait_consumer(&self) {
        if let Some(ref fence) = self.consumer_fence {
            fence.wait();
        }
    }

    /// Wait for the producer with a timeout.
    pub fn wait_producer_timeout(&self, timeout: Duration) -> Result<(), ()> {
        if let Some(ref fence) = self.producer_fence {
            fence.wait_timeout(timeout)
        } else {
            Ok(())
        }
    }

    /// Consume all produced items, returning them.
    pub fn consume_all(&self) -> Vec<String> {
        let mut items = self.items.lock().unwrap();
        std::mem::take(&mut *items)
    }

    /// Get the number of unconsumed items.
    pub fn pending_count(&self) -> usize {
        self.items.lock().unwrap().len()
    }
}

impl Default for ProducerConsumer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_signal_then_wait_succeeds() {
        let fence = Fence::new();
        assert!(!fence.is_signaled());

        fence.signal();
        assert!(fence.is_signaled());

        // Should return immediately since already signaled
        fence.wait();
    }

    #[test]
    fn test_signal_then_wait_cross_thread() {
        let fence = Fence::new();
        let fence_clone = fence.clone();

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            fence_clone.signal();
        });

        fence.wait();
        assert!(fence.is_signaled());
        handle.join().unwrap();
    }

    #[test]
    fn test_unsignaled_fence_blocks_simulated() {
        let fence = Fence::new();
        let fence_clone = fence.clone();

        // Spawn a thread that waits, then signal after a delay
        let handle = thread::spawn(move || {
            fence_clone.wait();
            true
        });

        // Thread should be blocked
        thread::sleep(Duration::from_millis(20));
        assert!(!handle.is_finished());

        // Signal to unblock
        fence.signal();
        let result = handle.join().unwrap();
        assert!(result);
    }

    #[test]
    fn test_fence_pool_reuses_fences() {
        let pool = FencePool::new();

        // First acquisition should create
        let f1 = pool.acquire();
        let id1 = f1.id();
        assert_eq!(pool.created_count(), 1);
        assert_eq!(pool.reused_count(), 0);

        // Release back to pool
        pool.release(f1);
        assert_eq!(pool.available(), 1);

        // Second acquisition should reuse
        let f2 = pool.acquire();
        assert_eq!(f2.id(), id1); // Same fence object reused
        assert_eq!(pool.created_count(), 1);
        assert_eq!(pool.reused_count(), 1);
        assert_eq!(pool.available(), 0);

        // Third acquisition should create new
        let f3 = pool.acquire();
        assert_ne!(f3.id(), id1);
        assert_eq!(pool.created_count(), 2);
        assert_eq!(pool.reused_count(), 1);
    }

    #[test]
    fn test_fence_pool_multiple_acquire_release() {
        let pool = FencePool::new();

        let mut ids = Vec::new();
        for _ in 0..5 {
            ids.push(pool.acquire());
        }
        assert_eq!(pool.created_count(), 5);

        for f in ids {
            pool.release(f);
        }
        assert_eq!(pool.available(), 5);

        // All should be reused
        for _ in 0..5 {
            pool.acquire();
        }
        assert_eq!(pool.reused_count(), 5);
        assert_eq!(pool.created_count(), 5);
    }

    #[test]
    fn test_timeout_detection() {
        let fence = Fence::new();

        // Not signaled, should timeout
        let result = fence.wait_timeout(Duration::from_millis(10));
        assert_eq!(result, Err(()));

        // Signal and try again
        fence.signal();
        let result = fence.wait_timeout(Duration::from_millis(10));
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn test_timeout_cross_thread() {
        let fence = Fence::new();
        let fence_clone = fence.clone();

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            fence_clone.signal();
        });

        // Should timeout before signal
        let result = fence.wait_timeout(Duration::from_millis(10));
        assert_eq!(result, Err(()));

        // Should succeed after waiting longer
        let result = fence.wait_timeout(Duration::from_millis(200));
        assert_eq!(result, Ok(()));

        handle.join().unwrap();
    }

    #[test]
    fn test_producer_consumer_ordering() {
        let mut pc = ProducerConsumer::new();
        let produce_fence = Fence::new();
        let consume_fence = Fence::new();

        pc.set_producer_fence(produce_fence.clone());
        pc.set_consumer_fence(consume_fence.clone());

        // Producer produces items
        pc.produce("item1");
        pc.produce("item2");
        pc.produce("item3");
        assert_eq!(pc.pending_count(), 3);

        // Signal production done
        pc.signal_producer_done();
        assert!(produce_fence.is_signaled());

        // Consumer waits for production, then consumes
        pc.wait_producer();
        let items = pc.consume_all();
        assert_eq!(items, vec!["item1", "item2", "item3"]);
        assert_eq!(pc.pending_count(), 0);

        // Signal consumption done
        pc.signal_consumer_done();
        assert!(consume_fence.is_signaled());
    }

    #[test]
    fn test_producer_consumer_cross_thread() {
        let produce_fence = Arc::new(Fence::new());
        let consume_fence = Arc::new(Fence::new());
        let items: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let items_producer = items.clone();
        let pf = produce_fence.clone();
        let producer = thread::spawn(move || {
            for i in 0..5 {
                items_producer.lock().unwrap().push(format!("item_{i}"));
            }
            pf.signal();
        });

        let items_consumer = items.clone();
        let cf = consume_fence.clone();
        let pf2 = produce_fence.clone();
        let consumer = thread::spawn(move || {
            pf2.wait();
            let consumed: Vec<String> = items_consumer.lock().unwrap().drain(..).collect();
            assert_eq!(consumed.len(), 5);
            cf.signal();
        });

        producer.join().unwrap();
        consumer.join().unwrap();
        assert!(produce_fence.is_signaled());
        assert!(consume_fence.is_signaled());
        assert!(items.lock().unwrap().is_empty());
    }

    #[test]
    fn test_fence_manager_create_and_signal() {
        let manager = FenceManager::new();
        let fence = manager.create_fence();
        let id = fence.id();

        assert_eq!(manager.active_count(), 1);
        assert_eq!(manager.get_status(id), Some(FenceStatus::Active));

        manager.signal_fence(id);
        assert_eq!(manager.get_status(id), Some(FenceStatus::Completed));
        assert_eq!(manager.completed_count(), 1);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_fence_manager_wait() {
        let manager = FenceManager::new();
        let fence = manager.create_fence();
        let id = fence.id();

        manager.signal_fence(id);
        let result = manager.wait_fence(id);
        assert!(result);
    }

    #[test]
    fn test_fence_manager_wait_nonexistent() {
        let manager = FenceManager::new();
        let result = manager.wait_fence(FenceId(9999));
        assert!(!result);
    }

    #[test]
    fn test_fence_manager_cleanup() {
        let manager = FenceManager::new();

        let f1 = manager.create_fence();
        let f2 = manager.create_fence();
        let _f3 = manager.create_fence();

        manager.signal_fence(f1.id());
        manager.signal_fence(f2.id());

        assert_eq!(manager.completed_count(), 2);
        assert_eq!(manager.active_count(), 1);

        let cleaned = manager.cleanup_completed();
        assert_eq!(cleaned, 2);
        assert_eq!(manager.total_count(), 1);
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_fence_manager_timeout() {
        let manager = FenceManager::new();
        let fence = manager.create_fence();
        let id = fence.id();

        let status = manager.wait_fence_timeout(id, Duration::from_millis(10));
        assert_eq!(status, FenceStatus::TimedOut);
        assert_eq!(manager.get_status(id), Some(FenceStatus::TimedOut));
    }

    #[test]
    fn test_fence_manager_wait_all() {
        let manager = FenceManager::new();
        let f1 = manager.create_fence();
        let f2 = manager.create_fence();
        let f3 = manager.create_fence();

        let mgr = Arc::new(manager);
        let mgr_clone = mgr.clone();
        let f1_id = f1.id();
        let f2_id = f2.id();

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            mgr_clone.signal_fence(f1_id);
            mgr_clone.signal_fence(f2_id);
        });

        // Wait for the first two (they'll be signaled by the thread)
        handle.join().unwrap();
        
        // Signal f3
        mgr.signal_fence(f3.id());
        
        mgr.wait_all();
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.completed_count(), 3);
    }

    #[test]
    fn test_fence_reset_and_reuse() {
        let fence = Fence::new();
        
        fence.signal();
        assert!(fence.is_signaled());
        
        fence.reset();
        assert!(!fence.is_signaled());
        
        // Should block again
        fence.signal();
        assert!(fence.is_signaled());
    }

    #[test]
    fn test_fence_labeled() {
        let fence = Fence::with_label("my_fence");
        assert_eq!(fence.label(), "my_fence");
    }

    #[test]
    fn test_fence_signaled_at() {
        let fence = Fence::new();
        assert!(fence.signaled_at().is_none());

        let before = Instant::now();
        fence.signal();
        let after = Instant::now();

        let signaled_at = fence.signaled_at().unwrap();
        assert!(signaled_at >= before);
        assert!(signaled_at <= after);
    }

    #[test]
    fn test_fence_pool_labeled() {
        let pool = FencePool::new();
        let f1 = pool.acquire_labeled("compute_done");
        assert_eq!(f1.label(), "compute_done");
    }

    #[test]
    fn test_fence_pool_stats() {
        let pool = FencePool::new();
        
        let f1 = pool.acquire();
        let f2 = pool.acquire();
        pool.release(f1);
        pool.release(f2);
        
        let f3 = pool.acquire();
        let f4 = pool.acquire();
        let f5 = pool.acquire();
        
        assert_eq!(pool.created_count(), 3); // 2 new initially, then 1 more
        assert_eq!(pool.reused_count(), 2);
        assert_eq!(pool.total_acquisitions(), 5);
        assert_eq!(pool.available(), 0);
        
        pool.release(f3);
        pool.release(f4);
        pool.release(f5);
        assert_eq!(pool.available(), 3);
    }

    #[test]
    fn test_fence_manager_with_pool() {
        let pool = Arc::new(FencePool::new());
        let manager = FenceManager::with_pool(pool.clone());

        let f1 = manager.create_fence();
        manager.signal_fence(f1.id());
        manager.cleanup_completed();

        // Fence should be back in pool
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_producer_consumer_timeout() {
        let mut pc = ProducerConsumer::new();
        let fence = Fence::new();
        pc.set_producer_fence(fence);

        // Producer hasn't signaled, should timeout
        let result = pc.wait_producer_timeout(Duration::from_millis(10));
        assert_eq!(result, Err(()));
    }

    #[test]
    fn test_multiple_waiters_on_one_fence() {
        let fence = Arc::new(Fence::new());
        let mut handles = Vec::new();

        for _ in 0..5 {
            let f = fence.clone();
            handles.push(thread::spawn(move || {
                f.wait();
                true
            }));
        }

        thread::sleep(Duration::from_millis(20));
        // All threads should be waiting
        for h in &handles {
            assert!(!h.is_finished());
        }

        fence.signal();

        for h in handles {
            assert!(h.join().unwrap());
        }
    }

    #[test]
    fn test_fence_clone_shares_state() {
        let fence = Fence::new();
        let clone = fence.clone();

        fence.signal();
        assert!(clone.is_signaled());
    }
}

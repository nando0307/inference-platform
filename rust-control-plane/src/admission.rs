use crate::error::Error;
use crate::identity::{Generation, ModelId, WorkerId, WorkerIdentity, WorkerState};
use crate::registry::{Counters, Registry};
use crate::routing::{self, Policy};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Failed,
    Canceled,
    Timeout,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoutingOutcomes {
    pub reserved: u64,
    pub server_busy: u64,
    pub not_ready: u64,
}
#[derive(Debug)]
struct State {
    registry: Registry,
    cursor: HashMap<ModelId, WorkerId>,
    credits: HashMap<ModelId, HashMap<WorkerId, i128>>,
    next_id: u64,
    outcomes: RoutingOutcomes,
}
#[derive(Debug, Clone)]
pub struct Admission {
    inner: Arc<Mutex<State>>,
    policy: Policy,
}
/// Sole owner of a reservation. Dropping one, including on panic or future abort, cancels it.
#[derive(Debug)]
pub struct Reservation {
    inner: Arc<Mutex<State>>,
    id: u64,
    worker: WorkerId,
    generation: Generation,
    released: bool,
}

fn count(c: &mut Counters, outcome: Outcome) {
    let n = match outcome {
        Outcome::Completed => &mut c.completed,
        Outcome::Failed => &mut c.failed,
        Outcome::Canceled => &mut c.canceled,
        Outcome::Timeout => &mut c.timeout,
    };
    *n += 1;
}
fn finish(
    state: &mut State,
    id: u64,
    worker: &WorkerId,
    generation: Generation,
    outcome: Outcome,
) -> bool {
    let Some(w) = state.registry.workers.get_mut(worker) else {
        return false;
    };
    if w.generation != generation {
        return false;
    }
    let active = w.active.remove(&id);
    let queued = if let Some(index) = w.queue.iter().position(|entry| *entry == id) {
        w.queue.remove(index);
        true
    } else {
        false
    };
    if !active && !queued {
        return false;
    }
    count(&mut w.counters, outcome);
    if active && w.state == WorkerState::Ready {
        while w.active.len() < w.identity.max_sequences {
            let Some(next) = w.queue.pop_front() else {
                break;
            };
            w.active.insert(next);
        }
    }
    true
}
impl Admission {
    pub fn new(
        identities: impl IntoIterator<Item = WorkerIdentity>,
        policy: Policy,
    ) -> Result<Self, Error> {
        Ok(Self {
            inner: Arc::new(Mutex::new(State {
                registry: Registry::new(identities)?,
                cursor: HashMap::new(),
                credits: HashMap::new(),
                next_id: 0,
                outcomes: RoutingOutcomes::default(),
            })),
            policy,
        })
    }
    /// Single mutex serializes readiness, selection, credit commit, slot ownership and counters.
    pub fn reserve(&self, model: &ModelId) -> Result<Reservation, Error> {
        let mut state = self.inner.lock().expect("admission state poisoned");
        if !state.registry.accepting {
            state.outcomes.not_ready += 1;
            return Err(Error::NotReady);
        }
        let candidates = state.registry.candidates(model);
        if candidates.is_empty() {
            if state.registry.ready(model) {
                let rejected = state
                    .registry
                    .workers
                    .values_mut()
                    .filter(|w| &w.identity.model == model && w.state == WorkerState::Ready)
                    .min_by_key(|w| (w.inflight(), w.identity.id.clone()))
                    .unwrap();
                rejected.counters.rejected += 1;
                state.outcomes.server_busy += 1;
                return Err(Error::ServerBusy);
            }
            state.outcomes.not_ready += 1;
            return Err(Error::NotReady);
        }
        let (worker, credits) = routing::select(
            &candidates,
            self.policy,
            state.cursor.get(model),
            state.credits.get(model).unwrap_or(&HashMap::new()),
        )?;
        let id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| Error::Config("Reservation ID exhausted".into()))?;
        state.next_id = id;
        let w = state
            .registry
            .workers
            .get_mut(&worker)
            .expect("selected worker exists");
        let generation = w.generation;
        if w.active.len() < w.identity.max_sequences {
            w.active.insert(id);
        } else {
            w.queue.push_back(id);
        }
        state.cursor.insert(model.clone(), worker.clone());
        state.credits.insert(model.clone(), credits);
        state.outcomes.reserved += 1;
        Ok(Reservation {
            inner: Arc::clone(&self.inner),
            id,
            worker,
            generation,
            released: false,
        })
    }
    /// Caller supplies the generation observed before external health/lifecycle work.
    /// Repeated or stale updates cannot resurrect a worker or an old reservation.
    pub fn set_state(
        &self,
        worker: &WorkerId,
        expected: Generation,
        next: WorkerState,
    ) -> Result<(), Error> {
        let mut state = self.inner.lock().expect("admission state poisoned");
        let w = state
            .registry
            .workers
            .get_mut(worker)
            .ok_or(Error::NotReady)?;
        if w.generation != expected {
            return Err(Error::StaleGeneration);
        }
        if (next == WorkerState::Ready
            && !matches!(w.state, WorkerState::Starting | WorkerState::Ready))
            || (next == WorkerState::Starting && w.state != WorkerState::Starting)
        {
            return Err(Error::StaleGeneration);
        }
        w.state = next;
        if next != WorkerState::Ready {
            while w.queue.pop_front().is_some() {
                count(&mut w.counters, Outcome::Failed);
            }
        }
        Ok(())
    }
    /// Fences old leases and pending health results before a new worker generation becomes ready.
    pub fn advance_generation(
        &self,
        worker: &WorkerId,
        expected: Generation,
    ) -> Result<Generation, Error> {
        let mut state = self.inner.lock().expect("admission state poisoned");
        let w = state
            .registry
            .workers
            .get_mut(worker)
            .ok_or(Error::NotReady)?;
        if w.generation != expected {
            return Err(Error::StaleGeneration);
        }
        let next = Generation(
            w.generation
                .0
                .checked_add(1)
                .ok_or_else(|| Error::Config("Worker generation exhausted".into()))?,
        );
        for _ in w.active.drain() {
            count(&mut w.counters, Outcome::Failed);
        }
        while w.queue.pop_front().is_some() {
            count(&mut w.counters, Outcome::Failed);
        }
        w.state = WorkerState::Starting;
        w.generation = next;
        Ok(next)
    }
    pub fn stop_accepting(&self) {
        self.inner
            .lock()
            .expect("admission state poisoned")
            .registry
            .accepting = false;
    }
    pub fn snapshot(&self) -> (RegistrySnapshot, RoutingOutcomes) {
        let state = self.inner.lock().expect("admission state poisoned");
        (
            RegistrySnapshot(
                state
                    .registry
                    .workers
                    .iter()
                    .map(|(id, w)| {
                        (
                            id.clone(),
                            WorkerSnapshot {
                                state: w.state,
                                generation: w.generation,
                                active: w.active.len(),
                                queued: w.queue.len(),
                                queue_ids: w.queue.iter().copied().collect(),
                                counters: w.counters.clone(),
                            },
                        )
                    })
                    .collect(),
            ),
            state.outcomes.clone(),
        )
    }
    pub fn ready(&self, models: impl IntoIterator<Item = ModelId>) -> bool {
        self.inner
            .lock()
            .expect("admission state poisoned")
            .registry
            .readiness(models)
    }
}
#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub state: WorkerState,
    pub generation: Generation,
    pub active: usize,
    pub queued: usize,
    pub queue_ids: Vec<u64>,
    pub counters: Counters,
}
#[derive(Debug, Clone)]
pub struct RegistrySnapshot(pub HashMap<WorkerId, WorkerSnapshot>);
impl Reservation {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn worker(&self) -> &WorkerId {
        &self.worker
    }
    pub fn generation(&self) -> Generation {
        self.generation
    }
    /// Verify current generation, readiness and active membership immediately before forwarding.
    pub fn dispatchable(&self) -> Result<(), Error> {
        let state = self.inner.lock().expect("admission state poisoned");
        let w = state.registry.get(&self.worker).ok_or(Error::NotReady)?;
        if w.generation != self.generation {
            return Err(Error::StaleGeneration);
        }
        if self.released || w.state != WorkerState::Ready || !w.active.contains(&self.id) {
            return Err(Error::NotReady);
        }
        Ok(())
    }
    pub fn release(&mut self, outcome: Outcome) -> bool {
        if self.released {
            return false;
        }
        self.released = true;
        let mut state = self.inner.lock().expect("admission state poisoned");
        finish(&mut state, self.id, &self.worker, self.generation, outcome)
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.release(Outcome::Canceled);
    }
}

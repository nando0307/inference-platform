use crate::error::Error;
use crate::identity::{Generation, ModelId, WorkerId, WorkerIdentity, WorkerState};
use crate::routing::Candidate;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counters {
    pub completed: u64,
    pub failed: u64,
    pub canceled: u64,
    pub timeout: u64,
    pub rejected: u64,
}
#[derive(Debug, Clone)]
pub struct WorkerRecord {
    pub identity: WorkerIdentity,
    pub state: WorkerState,
    pub generation: Generation,
    pub active: HashSet<u64>,
    pub queue: std::collections::VecDeque<u64>,
    pub counters: Counters,
}
impl WorkerRecord {
    pub fn inflight(&self) -> usize {
        self.active.len() + self.queue.len()
    }
    pub fn candidate(&self) -> Option<Candidate> {
        (self.state == WorkerState::Ready
            && (self.active.len() < self.identity.max_sequences
                || self.queue.len() < self.identity.queue_capacity))
            .then(|| Candidate {
                id: self.identity.id.clone(),
                inflight: self.inflight(),
                queued: self.queue.len(),
                weight: self.identity.weight.clone(),
            })
    }
}
#[derive(Debug, Default)]
pub struct Registry {
    pub(crate) workers: HashMap<WorkerId, WorkerRecord>,
    pub accepting: bool,
}
impl Registry {
    pub fn new(identities: impl IntoIterator<Item = WorkerIdentity>) -> Result<Self, Error> {
        let mut registry = Self {
            accepting: true,
            ..Self::default()
        };
        for identity in identities {
            if identity.max_sequences == 0 || identity.weight == 0u8.into() {
                return Err(Error::Config(
                    "Worker capacity and weight must be positive".into(),
                ));
            }
            if registry
                .workers
                .insert(
                    identity.id.clone(),
                    WorkerRecord {
                        identity,
                        state: WorkerState::Starting,
                        generation: Generation(0),
                        active: HashSet::new(),
                        queue: Default::default(),
                        counters: Counters::default(),
                    },
                )
                .is_some()
            {
                return Err(Error::DuplicateWorker);
            }
        }
        Ok(registry)
    }
    pub fn get(&self, id: &WorkerId) -> Option<&WorkerRecord> {
        self.workers.get(id)
    }
    pub fn candidates(&self, model: &ModelId) -> Vec<Candidate> {
        self.workers
            .values()
            .filter(|w| &w.identity.model == model)
            .filter_map(WorkerRecord::candidate)
            .collect()
    }
    pub fn ready(&self, model: &ModelId) -> bool {
        self.workers
            .values()
            .any(|w| &w.identity.model == model && w.state == WorkerState::Ready)
    }
    pub fn readiness(&self, models: impl IntoIterator<Item = ModelId>) -> bool {
        self.accepting && models.into_iter().all(|model| self.ready(&model))
    }
}

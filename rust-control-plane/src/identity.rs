use crate::config::Worker as ConfigWorker;
use num_bigint::BigUint;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(pub String);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Generation(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Starting,
    Ready,
    Unavailable,
    Stopped,
    CleanupFailed,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerIdentity {
    pub id: WorkerId,
    pub model: ModelId,
    pub host: String,
    pub gpu_uuid: String,
    pub cuda_device: usize,
    pub max_sequences: usize,
    pub queue_capacity: usize,
    pub weight: BigUint,
}
impl WorkerIdentity {
    pub fn from_config(worker: &ConfigWorker, queue_capacity: usize) -> Self {
        Self {
            id: WorkerId(worker.id.clone()),
            model: ModelId(worker.model.clone()),
            host: worker.host.clone(),
            gpu_uuid: worker.gpu_uuid.clone(),
            cuda_device: worker.cuda_device,
            max_sequences: worker.max_sequences,
            queue_capacity,
            weight: worker.capacity_weight.clone(),
        }
    }
}

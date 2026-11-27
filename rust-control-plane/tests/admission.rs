use rust_control_plane::{
    admission::{Admission, Outcome},
    config::{ServingConfig, SMALL},
    error::Error,
    identity::{Generation, ModelId, WorkerId, WorkerIdentity, WorkerState},
    routing::Policy,
};
use std::sync::{Arc, Barrier};

fn manager(policy: Policy, placement: &str) -> Admission {
    let config = ServingConfig::load(format!(
        "{}/../configs/serving-{placement}.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let ids = config
        .workers
        .iter()
        .map(|w| WorkerIdentity::from_config(w, config.queue_capacity_per_worker));
    let admission = Admission::new(ids, policy).unwrap();
    for worker in &config.workers {
        admission
            .set_state(
                &WorkerId(worker.id.clone()),
                Generation(0),
                WorkerState::Ready,
            )
            .unwrap();
    }
    admission
}
fn small() -> ModelId {
    ModelId(SMALL.into())
}
fn worker(a: &Admission, id: &str) -> rust_control_plane::admission::WorkerSnapshot {
    a.snapshot().0 .0[&WorkerId(id.into())].clone()
}
#[test]
fn bounds_fifo_errors_and_exactly_once() {
    let a = manager(Policy::LeastInflight, "d");
    let mut leases: Vec<_> = (0..4).map(|_| a.reserve(&small()).unwrap()).collect();
    assert_eq!(
        (worker(&a, "small-0").active, worker(&a, "small-0").queued),
        (2, 2)
    );
    assert_eq!(a.reserve(&small()).unwrap_err(), Error::ServerBusy);
    assert_eq!(
        a.reserve(&ModelId("other".into())).unwrap_err(),
        Error::NotReady
    );
    assert_eq!(worker(&a, "small-0").counters.rejected, 1);
    assert_eq!(a.snapshot().1.reserved, 4);
    assert_eq!(a.snapshot().1.server_busy, 1);
    assert_eq!(a.snapshot().1.not_ready, 1);
    let first_wait = leases[2].id();
    let second_wait = leases[3].id();
    assert_eq!(worker(&a, "small-0").queue_ids, [first_wait, second_wait]);
    assert!(leases[0].release(Outcome::Completed));
    assert!(!leases[0].release(Outcome::Failed));
    assert_eq!(worker(&a, "small-0").queue_ids, [second_wait]);
    assert!(leases[2].dispatchable().is_ok());
    drop(leases);
    let w = worker(&a, "small-0");
    assert_eq!((w.active, w.queued), (0, 0));
    assert_eq!(
        (w.counters.completed, w.counters.canceled, w.counters.failed),
        (1, 3, 0)
    );
}
#[test]
fn generation_fences_late_health_results_and_owners() {
    let a = manager(Policy::LeastInflight, "d");
    let mut old = a.reserve(&small()).unwrap();
    let next = a
        .advance_generation(&WorkerId("small-0".into()), Generation(0))
        .unwrap();
    assert_eq!(
        a.set_state(
            &WorkerId("small-0".into()),
            Generation(0),
            WorkerState::Ready
        ),
        Err(Error::StaleGeneration)
    );
    assert_eq!(old.dispatchable(), Err(Error::StaleGeneration));
    assert!(!old.release(Outcome::Completed));
    a.set_state(&WorkerId("small-0".into()), next, WorkerState::Ready)
        .unwrap();
    let new = a.reserve(&small()).unwrap();
    assert_ne!(old.id(), new.id());
    assert_eq!(new.generation(), next);
    drop(new);
    let w = worker(&a, "small-0");
    assert_eq!(
        (w.counters.failed, w.counters.canceled, w.counters.completed),
        (1, 1, 0)
    );
}
#[test]
fn concurrent_selection_reservation_and_abort_are_atomic() {
    let a = manager(Policy::RoundRobin, "b");
    let barrier = Arc::new(Barrier::new(16));
    let joins: Vec<_> = (0..16)
        .map(|_| {
            let a = a.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                a.reserve(&small())
            })
        })
        .collect();
    let leases: Vec<_> = joins
        .into_iter()
        .filter_map(|j| j.join().unwrap().ok())
        .collect();
    assert_eq!(leases.len(), 8);
    for id in ["small-0", "small-1"] {
        let w = worker(&a, id);
        assert_eq!((w.active, w.queued), (2, 2));
    }
    drop(leases); // Models future cancellation / task abort without explicit cleanup.
    for id in ["small-0", "small-1"] {
        let w = worker(&a, id);
        assert_eq!((w.active, w.queued, w.counters.canceled), (0, 0, 4));
    }
}
#[test]
fn weighted_admission_sequence_and_model_isolation() {
    let config = ServingConfig::load(format!(
        "{}/../configs/serving-b.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let identities = config.workers.iter().map(|w| {
        let mut id = WorkerIdentity::from_config(w, config.queue_capacity_per_worker);
        if id.id.0 == "small-0" {
            id.weight = 3;
        }
        id
    });
    let a = Admission::new(identities, Policy::WeightedCapacity).unwrap();
    for w in &config.workers {
        a.set_state(&WorkerId(w.id.clone()), Generation(0), WorkerState::Ready)
            .unwrap();
    }
    let ids: Vec<_> = (0..4)
        .map(|_| {
            let lease = a.reserve(&small()).unwrap();
            let id = lease.worker().0.clone();
            drop(lease);
            id
        })
        .collect();
    assert_eq!(ids, ["small-0", "small-0", "small-1", "small-0"]);
    let large = a
        .reserve(&ModelId(rust_control_plane::config::LARGE.into()))
        .unwrap();
    assert_eq!(large.worker().0, "large-0");
    assert_eq!(a.snapshot().1.reserved, 5);
}
#[test]
fn release_racing_generation_change_counts_once() {
    for _ in 0..100 {
        let a = manager(Policy::LeastInflight, "d");
        let mut lease = a.reserve(&small()).unwrap();
        let clone = a.clone();
        let barrier = Arc::new(Barrier::new(2));
        let twin = barrier.clone();
        let task = std::thread::spawn(move || {
            twin.wait();
            clone
                .advance_generation(&WorkerId("small-0".into()), Generation(0))
                .unwrap();
        });
        barrier.wait();
        lease.release(Outcome::Completed);
        task.join().unwrap();
        let w = worker(&a, "small-0");
        assert_eq!(w.active + w.queued, 0);
        assert_eq!(w.counters.completed + w.counters.failed, 1);
        drop(lease);
        assert_eq!(
            worker(&a, "small-0").counters.completed + worker(&a, "small-0").counters.failed,
            1
        );
    }
}
#[test]
fn unavailable_drains_queue_without_double_counting() {
    let a = manager(Policy::LeastInflight, "d");
    let leases: Vec<_> = (0..4).map(|_| a.reserve(&small()).unwrap()).collect();
    a.set_state(
        &WorkerId("small-0".into()),
        Generation(0),
        WorkerState::Unavailable,
    )
    .unwrap();
    assert_eq!(worker(&a, "small-0").counters.failed, 2);
    assert_eq!(leases[2].dispatchable(), Err(Error::NotReady));
    for forbidden in [WorkerState::Ready, WorkerState::Starting] {
        assert_eq!(
            a.set_state(&WorkerId("small-0".into()), Generation(0), forbidden),
            Err(Error::StaleGeneration)
        );
    }
    drop(leases);
    let w = worker(&a, "small-0");
    assert_eq!((w.counters.failed, w.counters.canceled), (2, 2));
}

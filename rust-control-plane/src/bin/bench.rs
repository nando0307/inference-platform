//! CPU-only deterministic control-plane microbenchmarks; emits every measured latency (ns).
use rust_control_plane::{
    admission::{Admission, Outcome},
    config::{ServingConfig, SMALL},
    identity::{Generation, ModelId, WorkerId, WorkerIdentity, WorkerState},
    routing::{self, Candidate, Policy},
};
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

fn admission() -> Admission {
    let config = ServingConfig::load(format!(
        "{}/../configs/serving-b.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let a = Admission::new(
        config
            .workers
            .iter()
            .map(|w| WorkerIdentity::from_config(w, config.queue_capacity_per_worker)),
        Policy::LeastInflight,
    )
    .unwrap();
    for w in &config.workers {
        a.set_state(&WorkerId(w.id.clone()), Generation(0), WorkerState::Ready)
            .unwrap();
    }
    a
}
fn sample(kind: &str, iterations: usize) -> (Vec<u64>, u64, u64, u64) {
    let mut ns = Vec::with_capacity(iterations);
    let mut rejected = 0;
    let mut capacity = 0;
    let model = ModelId(SMALL.into());
    let a = admission();
    let c = [
        Candidate {
            id: WorkerId("small-0".into()),
            inflight: 0,
            queued: 0,
            weight: 1,
        },
        Candidate {
            id: WorkerId("small-1".into()),
            inflight: 1,
            queued: 0,
            weight: 1,
        },
    ];
    let wall = Instant::now();
    for _ in 0..iterations {
        let start = Instant::now();
        match kind {
            "selector" => {
                std::hint::black_box(
                    routing::select(&c, Policy::LeastInflight, None, &HashMap::new()).unwrap(),
                );
            }
            "reservation" => {
                let lease = a.reserve(&model).unwrap();
                ns.push(start.elapsed().as_nanos() as u64);
                drop(lease);
                continue;
            }
            "synthetic" => {
                let mut leases = Vec::new();
                for _ in 0..9 {
                    match a.reserve(&model) {
                        Ok(lease) => leases.push(lease),
                        Err(_) => rejected += 1,
                    }
                }
                capacity = capacity.max(leases.len() as u64);
                for mut lease in leases {
                    // FIFO promotion makes every lease dispatchable in reservation order.
                    lease.dispatchable().unwrap();
                    lease.release(Outcome::Completed);
                }
            }
            _ => panic!("unknown workload"),
        }
        ns.push(start.elapsed().as_nanos() as u64);
    }
    (ns, rejected, capacity, wall.elapsed().as_nanos() as u64)
}
fn main() {
    let kind = std::env::args().nth(1).expect("workload");
    let warmup: usize = std::env::args().nth(2).expect("warmup").parse().unwrap();
    let iterations: usize = std::env::args()
        .nth(3)
        .expect("iterations")
        .parse()
        .unwrap();
    let repeats: usize = std::env::args().nth(4).expect("repeats").parse().unwrap();
    sample(&kind, warmup);
    for repeat in 0..repeats {
        let (samples_ns, rejected, capacity, elapsed_ns) = sample(&kind, iterations);
        println!(
            "{}",
            json!({"workload":kind,"repeat":repeat,"samples_ns":samples_ns,"elapsed_ns":elapsed_ns,"rejected":rejected,"capacity":capacity})
        );
    }
}

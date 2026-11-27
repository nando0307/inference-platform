use num_bigint::{BigInt, BigUint};
use rust_control_plane::{
    admission::{Admission, Outcome, Reservation},
    config::ServingConfig,
    identity::{Generation, ModelId, WorkerId, WorkerIdentity, WorkerState},
    routing::{select, Candidate, Policy},
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead};

fn evaluate(raw: &Value) -> Value {
    match raw["kind"].as_str().unwrap() {
        "config" => json!({"valid": ServingConfig::validate(&raw["raw"]).is_ok()}),
        "select" => {
            let candidates: Vec<_> = raw["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| Candidate {
                    id: WorkerId(c["id"].as_str().unwrap().into()),
                    inflight: c["inflight"].as_u64().unwrap() as usize,
                    queued: c["queued"].as_u64().unwrap() as usize,
                    weight: BigUint::parse_bytes(c["weight"].to_string().as_bytes(), 10).unwrap(),
                })
                .collect();
            let cursor = raw["cursor"].as_str().map(|s| WorkerId(s.into()));
            let credits: HashMap<_, _> = raw["credits"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (WorkerId(k.clone()), BigInt::from(v.as_i64().unwrap())))
                .collect();
            match select(
                &candidates,
                Policy::try_from(raw["policy"].as_str().unwrap()).unwrap(),
                cursor.as_ref(),
                &credits,
            ) {
                Ok((id, next)) => {
                    json!({"chosen": id.0, "credits": next.into_iter().map(|(k, v)| (k.0, v)).collect::<HashMap<_, _>>() })
                }
                Err(_) => json!({"error": "no_eligible"}),
            }
        }
        "trace" => {
            let config = ServingConfig::load(format!(
                "{}/../configs/serving-{}.json",
                env!("CARGO_MANIFEST_DIR"),
                raw["placement"].as_str().unwrap()
            ))
            .unwrap();
            let a = Admission::new(
                config
                    .workers
                    .iter()
                    .map(|w| WorkerIdentity::from_config(w, config.queue_capacity_per_worker)),
                Policy::try_from(raw["policy"].as_str().unwrap()).unwrap(),
            )
            .unwrap();
            for w in &config.workers {
                a.set_state(&WorkerId(w.id.clone()), Generation(0), WorkerState::Ready)
                    .unwrap();
            }
            let mut leases: HashMap<String, Reservation> = HashMap::new();
            let mut results = Vec::new();
            for step in raw["actions"].as_array().unwrap() {
                let name = step["name"].as_str().unwrap();
                match step["op"].as_str().unwrap() {
                    "reserve" => {
                        match a.reserve(&ModelId(step["model"].as_str().unwrap().into())) {
                            Ok(lease) => {
                                results.push(json!({"name": name, "worker": lease.worker().0}));
                                leases.insert(name.into(), lease);
                            }
                            Err(e) => results.push(json!({"name": name, "error": e.code()})),
                        }
                    }
                    "release" => {
                        let mut lease = leases.remove(name).unwrap();
                        lease.release(Outcome::Completed);
                        results.push(json!({"name": name, "released": true}));
                    }
                    "cancel" => {
                        leases.remove(name).unwrap();
                        results.push(json!({"name": name, "canceled": true}));
                    }
                    _ => unreachable!(),
                }
            }
            drop(leases);
            let (snap, outcomes) = a.snapshot();
            let counts: HashMap<_, _> = snap.0.into_iter().map(|(k, v)| (k.0, json!({"active":v.active,"queued":v.queued,"completed":v.counters.completed,"canceled":v.counters.canceled,"rejected":v.counters.rejected}))).collect();
            json!({"results": results, "workers": counts, "outcomes": {"reserved":outcomes.reserved,"server_busy":outcomes.server_busy,"not_ready":outcomes.not_ready}})
        }
        _ => unreachable!(),
    }
}
fn main() {
    for line in io::stdin().lock().lines() {
        let raw: Value = serde_json::from_str(&line.unwrap()).unwrap();
        println!("{}", evaluate(&raw));
    }
}

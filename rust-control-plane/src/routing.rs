use crate::error::Error;
use crate::identity::WorkerId;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    LeastInflight,
    LeastQueueDepth,
    RoundRobin,
    WeightedCapacity,
}
impl TryFrom<&str> for Policy {
    type Error = Error;
    fn try_from(value: &str) -> Result<Self, Error> {
        match value {
            "least-inflight" => Ok(Self::LeastInflight),
            "least-queue-depth" => Ok(Self::LeastQueueDepth),
            "round-robin" => Ok(Self::RoundRobin),
            "weighted-capacity" => Ok(Self::WeightedCapacity),
            _ => Err(Error::Config("Unknown routing policy".into())),
        }
    }
}
#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: WorkerId,
    pub inflight: usize,
    pub queued: usize,
    pub weight: u64,
}

/// Pure selection; credit state is returned separately and committed only on reservation.
pub fn select(
    candidates: &[Candidate],
    policy: Policy,
    cursor: Option<&WorkerId>,
    credits: &HashMap<WorkerId, i128>,
) -> Result<(WorkerId, HashMap<WorkerId, i128>), Error> {
    if candidates.is_empty() {
        return Err(Error::NotReady);
    }
    let mut ordered: Vec<_> = candidates.iter().collect();
    ordered.sort_by(|a, b| a.id.cmp(&b.id));
    let mut next = credits.clone();
    let chosen = match policy {
        Policy::LeastInflight => ordered
            .iter()
            .min_by_key(|c| (c.inflight, &c.id))
            .unwrap()
            .id
            .clone(),
        Policy::LeastQueueDepth => ordered
            .iter()
            .min_by_key(|c| (c.queued, &c.id))
            .unwrap()
            .id
            .clone(),
        Policy::RoundRobin => ordered
            .iter()
            .find(|c| cursor.is_none_or(|last| c.id > *last))
            .unwrap_or(&ordered[0])
            .id
            .clone(),
        Policy::WeightedCapacity => {
            // Credits are per model; only currently eligible workers accrue credit.
            for c in &ordered {
                *next.entry(c.id.clone()).or_default() += i128::from(c.weight);
            }
            let selected = ordered
                .iter()
                .max_by(|a, b| next[&a.id].cmp(&next[&b.id]).then_with(|| b.id.cmp(&a.id)))
                .unwrap();
            *next.get_mut(&selected.id).unwrap() -=
                ordered.iter().map(|c| i128::from(c.weight)).sum::<i128>();
            selected.id.clone()
        }
    };
    Ok((chosen, next))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_policies_and_credits() {
        let c = vec![
            Candidate {
                id: WorkerId("b".into()),
                inflight: 0,
                queued: 0,
                weight: 1,
            },
            Candidate {
                id: WorkerId("a".into()),
                inflight: 0,
                queued: 0,
                weight: 3,
            },
        ];
        for policy in [
            Policy::LeastInflight,
            Policy::LeastQueueDepth,
            Policy::RoundRobin,
            Policy::WeightedCapacity,
        ] {
            assert_eq!(select(&c, policy, None, &HashMap::new()).unwrap().0 .0, "a");
        }
        let mut credits = HashMap::new();
        let got: Vec<_> = (0..8)
            .map(|_| {
                let (id, next) = select(&c, Policy::WeightedCapacity, None, &credits).unwrap();
                credits = next;
                id.0
            })
            .collect();
        assert_eq!(got, ["a", "a", "b", "a", "a", "a", "b", "a"]);
        assert!(credits.values().all(|value| *value == 0));
    }
}

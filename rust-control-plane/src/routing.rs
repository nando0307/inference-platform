use crate::error::Error;
use crate::identity::WorkerId;
use num_bigint::{BigInt, BigUint};
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
    pub weight: BigUint,
}

/// Pure selection; credit state is returned separately and committed only on reservation.
pub fn select(
    candidates: &[Candidate],
    policy: Policy,
    cursor: Option<&WorkerId>,
    credits: &HashMap<WorkerId, BigInt>,
) -> Result<(WorkerId, HashMap<WorkerId, BigInt>), Error> {
    if candidates.is_empty() {
        return Err(Error::NotReady);
    }
    let mut ordered: Vec<_> = candidates.iter().collect();
    ordered.sort_by(|a, b| a.id.cmp(&b.id));
    let mut next = if policy == Policy::WeightedCapacity {
        ordered
            .iter()
            .map(|c| {
                (
                    c.id.clone(),
                    credits.get(&c.id).cloned().unwrap_or_default(),
                )
            })
            .collect()
    } else {
        credits.clone()
    };
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
                *next.entry(c.id.clone()).or_default() += BigInt::from(c.weight.clone());
            }
            let selected = ordered
                .iter()
                .max_by(|a, b| next[&a.id].cmp(&next[&b.id]).then_with(|| b.id.cmp(&a.id)))
                .unwrap();
            *next.get_mut(&selected.id).unwrap() -= ordered
                .iter()
                .map(|c| BigInt::from(c.weight.clone()))
                .sum::<BigInt>();
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
                weight: BigUint::from(1u8),
            },
            Candidate {
                id: WorkerId("a".into()),
                inflight: 0,
                queued: 0,
                weight: BigUint::from(3u8),
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
        assert!(credits.values().all(|value| *value == BigInt::from(0)));
        let huge = BigUint::parse_bytes(b"18446744073709551616", 10).unwrap();
        let candidates = vec![Candidate {
            id: WorkerId("large".into()),
            inflight: 0,
            queued: 0,
            weight: huge.clone(),
        }];
        let (chosen, credits) =
            select(&candidates, Policy::WeightedCapacity, None, &HashMap::new()).unwrap();
        assert_eq!(chosen.0, "large");
        assert_eq!(credits[&chosen], BigInt::from(0));
    }

    #[test]
    fn weighted_credits_follow_eligibility() {
        let a = Candidate {
            id: WorkerId("a".into()),
            inflight: 0,
            queued: 0,
            weight: 1u8.into(),
        };
        let b = Candidate {
            id: WorkerId("b".into()),
            inflight: 0,
            queued: 0,
            weight: 3u8.into(),
        };
        let both = [a.clone(), b.clone()];
        let (chosen, credits) =
            select(&both, Policy::WeightedCapacity, None, &HashMap::new()).unwrap();
        assert_eq!(chosen, b.id);
        let (chosen, credits) = select(&[a], Policy::WeightedCapacity, None, &credits).unwrap();
        assert_eq!(chosen.0, "a");
        assert_eq!(
            credits,
            HashMap::from([(WorkerId("a".into()), BigInt::from(1))])
        );
        let (chosen, credits) = select(&both, Policy::WeightedCapacity, None, &credits).unwrap();
        assert_eq!(chosen.0, "b");
        let (chosen, _) = select(&both, Policy::WeightedCapacity, None, &credits).unwrap();
        assert_eq!(chosen.0, "a");
    }
}

use crate::error::Error;
use num_bigint::BigUint;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use url::Url;

pub const SMALL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
pub const LARGE: &str = "Qwen/Qwen3-1.7B";
const SMALL_REV: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
const LARGE_REV: &str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e";

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub id: String,
    pub revision: String,
    pub context_tokens: u64,
}
#[derive(Debug, Clone, Deserialize)]
pub struct Host {
    pub gpu_uuids: Vec<String>,
}
#[derive(Debug, Clone, Deserialize)]
pub struct Worker {
    pub id: String,
    pub model: String,
    pub host: String,
    pub cuda_device: usize,
    pub gpu_uuid: String,
    pub endpoint: String,
    pub memory_utilization: f64,
    pub max_sequences: usize,
    pub startup_timeout_s: f64,
    pub request_timeout_s: f64,
    pub service_token_env: String,
    #[serde(default = "one", deserialize_with = "parse_weight")]
    pub capacity_weight: BigUint,
}
fn parse_weight<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<BigUint, D::Error> {
    let value = String::deserialize(deserializer)?;
    value.parse().map_err(serde::de::Error::custom)
}
fn one() -> BigUint {
    BigUint::from(1u8)
}
#[derive(Debug, Clone)]
pub struct ServingConfig {
    pub schema_version: u64,
    pub placement: String,
    pub models: HashMap<String, Model>,
    pub hosts: HashMap<String, Host>,
    pub workers: Vec<Worker>,
    pub queue_capacity_per_worker: usize,
    pub health_interval_s: f64,
    pub health_timeout_s: f64,
    pub policy: String,
}
fn invalid(message: &str) -> Error {
    Error::Config(message.into())
}
fn object<'a>(
    value: &'a Value,
    required: &[&str],
    optional: &[&str],
    msg: &str,
) -> Result<&'a Map<String, Value>, Error> {
    let map = value.as_object().ok_or_else(|| invalid(msg))?;
    if required.iter().any(|k| !map.contains_key(*k))
        || map
            .keys()
            .any(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
    {
        return Err(invalid(msg));
    }
    Ok(map)
}
fn number(value: &Value, name: &str, max: f64) -> Result<f64, Error> {
    let n = value.as_f64().ok_or_else(|| invalid(name))?;
    if !n.is_finite() || n <= 0.0 || n > max {
        return Err(invalid(name));
    }
    Ok(n)
}
fn integer(value: &Value, n: u64, name: &str) -> Result<(), Error> {
    if value.as_u64() != Some(n) {
        Err(invalid(name))
    } else {
        Ok(())
    }
}
fn gpu_uuid(s: &str) -> bool {
    s.strip_prefix("GPU-").is_some_and(|tail| {
        tail.len() == 36 && tail.bytes().all(|b| b == b'-' || b.is_ascii_hexdigit())
    })
}
fn token_env(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c == '_' || unicode_ident::is_xid_start(c))
        && chars.all(|c| c == '_' || unicode_ident::is_xid_continue(c))
}

impl ServingConfig {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let raw: Value = serde_json::from_str(text).map_err(|e| invalid(&e.to_string()))?;
        Self::validate(&raw)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(|e| invalid(&e.to_string()))?;
        Self::parse(&text)
    }
    pub fn validate(raw: &Value) -> Result<Self, Error> {
        let root = object(
            raw,
            &[
                "schema_version",
                "placement",
                "models",
                "hosts",
                "workers",
                "routing",
            ],
            &[],
            "Serving configuration has unsupported schema or fields.",
        )?;
        integer(
            &root["schema_version"],
            1,
            "Unsupported serving configuration schema.",
        )?;
        let placement = root["placement"]
            .as_str()
            .filter(|s| ["A", "B", "C", "D"].contains(s))
            .ok_or_else(|| invalid("placement must be A, B, C, or D."))?;
        let models_raw = object(
            &root["models"],
            &["small", "large"],
            &[],
            "models must define exactly small and large.",
        )?;
        let mut models = HashMap::new();
        for (name, id, rev) in [("small", SMALL, SMALL_REV), ("large", LARGE, LARGE_REV)] {
            let row = object(
                &models_raw[name],
                &["id", "revision", "dtype", "context_tokens"],
                &[],
                "model schema",
            )?;
            if row["id"] != id || row["revision"] != rev || row["dtype"] != "float16" {
                return Err(invalid(
                    "model must use its pinned allowlisted model, revision and dtype.",
                ));
            }
            integer(
                &row["context_tokens"],
                2048,
                "context_tokens must be the supported 2048-token policy.",
            )?;
            models.insert(
                name.into(),
                Model {
                    id: id.into(),
                    revision: rev.into(),
                    context_tokens: 2048,
                },
            );
        }
        let hosts_raw = root["hosts"]
            .as_object()
            .filter(|m| !m.is_empty())
            .ok_or_else(|| invalid("At least one host is required."))?;
        let mut hosts = HashMap::new();
        let mut uuids = HashSet::new();
        for (id, value) in hosts_raw {
            let row = object(
                value,
                &["gpu_uuids"],
                &[],
                "Each host requires a nonempty list of physical GPU UUIDs.",
            )?;
            let list = row["gpu_uuids"]
                .as_array()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    invalid("Each host requires a nonempty list of physical GPU UUIDs.")
                })?;
            let mut parsed = Vec::new();
            for uuid in list {
                let uuid = uuid.as_str().filter(|s| gpu_uuid(s)).ok_or_else(|| {
                    invalid("Each host requires a nonempty list of physical GPU UUIDs.")
                })?;
                if !uuids.insert(uuid.to_string()) {
                    return Err(invalid(
                        "Physical GPU UUIDs must be unique within this static topology.",
                    ));
                }
                parsed.push(uuid.to_string());
            }
            if id.is_empty() {
                return Err(invalid(
                    "Each host requires a nonempty list of physical GPU UUIDs.",
                ));
            }
            hosts.insert(id.clone(), Host { gpu_uuids: parsed });
        }
        let rows = root["workers"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("At least one worker is required."))?;
        let mut workers = Vec::new();
        let (mut ids, mut origins, mut slots, mut envs) = (
            HashSet::new(),
            HashSet::new(),
            HashSet::new(),
            HashSet::new(),
        );
        let mut budgets: HashMap<(&str, &str), f64> = HashMap::new();
        for value in rows {
            let row = object(
                value,
                &[
                    "id",
                    "model",
                    "host",
                    "cuda_device",
                    "gpu_uuid",
                    "endpoint",
                    "memory_utilization",
                    "max_sequences",
                    "startup_timeout_s",
                    "request_timeout_s",
                    "service_token_env",
                ],
                &["capacity_weight"],
                "Each worker must use the complete supported worker schema.",
            )?;
            let id = row["id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| invalid("Worker IDs must be nonempty strings and unique."))?;
            if !ids.insert(id) {
                return Err(invalid("Worker IDs must be nonempty strings and unique."));
            }
            let model = row["model"]
                .as_str()
                .filter(|s| [SMALL, LARGE].contains(s))
                .ok_or_else(|| invalid("Worker references an unknown model."))?;
            let host = row["host"]
                .as_str()
                .and_then(|s| hosts.get_key_value(s))
                .ok_or_else(|| {
                    invalid("Worker references an unknown host or local CUDA ordinal.")
                })?;
            let ordinal = row["cuda_device"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n < host.1.gpu_uuids.len())
                .ok_or_else(|| {
                    invalid("Worker references an unknown host or local CUDA ordinal.")
                })?;
            let uuid = row["gpu_uuid"]
                .as_str()
                .filter(|s| *s == host.1.gpu_uuids[ordinal])
                .ok_or_else(|| invalid("Worker CUDA ordinal and physical GPU UUID disagree."))?;
            if !slots.insert((host.0.as_str(), uuid, model)) {
                return Err(invalid(
                    "A model may have only one worker per physical GPU.",
                ));
            }
            let endpoint = row["endpoint"]
                .as_str()
                .ok_or_else(|| invalid("Worker endpoint must be a string."))?;
            let url = Url::parse(endpoint).map_err(|_| {
                invalid(
                    "Worker endpoint must be an explicit credential-free HTTP origin with a port.",
                )
            })?;
            let explicit_port = endpoint
                .split('/')
                .nth(2)
                .and_then(|authority| authority.rsplit_once(':'))
                .is_some_and(|(_, port)| !port.is_empty());
            let raw_path = endpoint
                .split_once("://")
                .and_then(|(_, rest)| rest.split(['?', '#']).next())
                .and_then(|rest| rest.split_once('/').map(|(_, path)| path))
                .unwrap_or("");
            if url.scheme() != "http"
                || url.host_str().is_none()
                || !explicit_port
                || url.port_or_known_default() == Some(0)
                || !url.username().is_empty()
                || url.password().is_some()
                || !raw_path.is_empty()
                || endpoint.contains('\\')
                || !["", "/"].contains(&url.path())
                || url.query().is_some_and(|q| !q.is_empty())
                || url.fragment().is_some_and(|f| !f.is_empty())
            {
                return Err(invalid(
                    "Worker endpoint must be an explicit credential-free HTTP origin with a port.",
                ));
            }
            let host_name = url.host_str().unwrap();
            let normalized = if host_name == "localhost" {
                "127.0.0.1"
            } else {
                host_name
            };
            if !origins.insert((normalized.to_string(), url.port_or_known_default().unwrap())) {
                return Err(invalid("Worker endpoints must be unique."));
            }
            let env = row["service_token_env"]
                .as_str()
                .filter(|s| token_env(s))
                .ok_or_else(|| {
                    invalid(
                        "Each worker requires a distinct service-token environment variable name.",
                    )
                })?;
            if !envs.insert(env) {
                return Err(invalid(
                    "Each worker requires a distinct service-token environment variable name.",
                ));
            }
            let memory = number(&row["memory_utilization"], "memory_utilization", 0.9)?;
            *budgets.entry((host.0, uuid)).or_default() += memory;
            integer(&row["max_sequences"], 2, "max_sequences must be 2.")?;
            number(&row["startup_timeout_s"], "startup_timeout_s", 360.0)?;
            number(&row["request_timeout_s"], "request_timeout_s", 180.0)?;
            if let Some(weight) = row.get("capacity_weight") {
                if weight
                    .as_number()
                    .and_then(|n| n.to_string().parse::<BigUint>().ok())
                    .is_none_or(|n| n == BigUint::default())
                {
                    return Err(invalid("capacity_weight must be a positive integer."));
                }
            }
            let mut worker_value = value.clone();
            if let Some(weight) = row.get("capacity_weight") {
                worker_value["capacity_weight"] = Value::String(weight.to_string());
            }
            workers.push(
                serde_json::from_value::<Worker>(worker_value)
                    .map_err(|e| invalid(&e.to_string()))?,
            );
        }
        if !workers.iter().any(|w| w.model == SMALL) || !workers.iter().any(|w| w.model == LARGE) {
            return Err(invalid(
                "Each configured model must have at least one worker.",
            ));
        }
        let expected: &[(&str, &str, usize)] = match placement {
            "A" => &[("small-0", SMALL, 0), ("large-1", LARGE, 1)],
            "B" => &[
                ("small-0", SMALL, 0),
                ("large-0", LARGE, 0),
                ("small-1", SMALL, 1),
            ],
            "C" => &[
                ("small-0", SMALL, 0),
                ("large-0", LARGE, 0),
                ("large-1", LARGE, 1),
            ],
            _ => &[("small-0", SMALL, 0), ("large-0", LARGE, 0)],
        };
        let actual: HashSet<_> = workers
            .iter()
            .map(|w| (w.model.as_str(), w.cuda_device))
            .collect();
        if hosts.len() != 1
            || hosts.values().next().unwrap().gpu_uuids.len() != 2
            || workers.len() != expected.len()
            || actual
                != expected
                    .iter()
                    .map(|(_, model, ordinal)| (*model, *ordinal))
                    .collect()
            || ids
                .iter()
                .any(|id| !expected.iter().any(|(name, _, _)| id == name))
        {
            return Err(invalid("Worker topology does not match placement."));
        }
        if budgets.values().any(|v| *v > 1.0) {
            return Err(invalid(
                "Declared per-process memory budgets exceed the per-GPU policy cap.",
            ));
        }
        let routing = object(
            &root["routing"],
            &[
                "queue_capacity_per_worker",
                "health_interval_s",
                "health_timeout_s",
            ],
            &["policy"],
            "routing must use a supported policy and schema.",
        )?;
        let policy = routing
            .get("policy")
            .map(Value::as_str)
            .unwrap_or(Some("least-inflight"))
            .filter(|s| {
                [
                    "least-inflight",
                    "least-queue-depth",
                    "round-robin",
                    "weighted-capacity",
                ]
                .contains(s)
            })
            .ok_or_else(|| invalid("routing must use a supported policy and schema."))?;
        if routing["queue_capacity_per_worker"].as_f64() != Some(2.0) {
            return Err(invalid("queue_capacity_per_worker must be 2."));
        }
        let interval = number(&routing["health_interval_s"], "health_interval_s", 60.0)?;
        let timeout = number(&routing["health_timeout_s"], "health_timeout_s", 30.0)?;
        if timeout > interval {
            return Err(invalid(
                "health timeout may not exceed the health interval.",
            ));
        }
        Ok(Self {
            schema_version: 1,
            placement: placement.into(),
            models,
            hosts,
            workers,
            queue_capacity_per_worker: 2,
            health_interval_s: interval,
            health_timeout_s: timeout,
            policy: policy.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placements_and_invalid_schema() {
        for p in ["a", "b", "c", "d"] {
            let file = format!("{}/../configs/serving-{p}.json", env!("CARGO_MANIFEST_DIR"));
            let text = std::fs::read_to_string(file).unwrap();
            let raw: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(
                ServingConfig::validate(&raw)
                    .unwrap()
                    .placement
                    .to_lowercase(),
                p
            );
            let mut bad = raw.clone();
            bad["schema_version"] = Value::Bool(true);
            assert!(ServingConfig::validate(&bad).is_err());
            bad = raw.clone();
            bad["workers"][0]["capacity_weight"] = Value::Bool(true);
            assert!(ServingConfig::validate(&bad).is_err());
            for weight in [
                "18446744073709551616",
                "100000000000000000000000000000000000000000000000000",
            ] {
                bad = raw.clone();
                bad["workers"][0]["capacity_weight"] = serde_json::from_str(weight).unwrap();
                assert_eq!(
                    ServingConfig::validate(&bad).unwrap().workers[0]
                        .capacity_weight
                        .to_string(),
                    weight
                );
            }
        }
    }
}

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// JSON-RPC 2.0 Request identifier (String, Number, or Null)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RpcId {
    Number(i64),
    String(String),
    Null,
}

impl Serialize for RpcId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            RpcId::Number(n) => serializer.serialize_i64(*n),
            RpcId::String(s) => serializer.serialize_str(s),
            RpcId::Null => serializer.serialize_none(),
        }
    }
}

impl<'de> Deserialize<'de> for RpcId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let val = serde_json::Value::deserialize(deserializer)?;
        match val {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(RpcId::Number(i))
                } else if let Some(u) = n.as_u64() {
                    Ok(RpcId::Number(u as i64))
                } else {
                    Ok(RpcId::String(n.to_string()))
                }
            }
            serde_json::Value::String(s) => Ok(RpcId::String(s)),
            serde_json::Value::Null => Ok(RpcId::Null),
            _ => Err(serde::de::Error::custom(
                "ID must be string, number, or null",
            )),
        }
    }
}

/// JSON-RPC 2.0 Error object
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl RpcError {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;

    pub fn parse_error(details: impl Into<String>) -> Self {
        Self {
            code: Self::PARSE_ERROR,
            message: "Parse error".to_string(),
            data: Some(serde_json::Value::String(details.into())),
        }
    }

    pub fn invalid_request(details: impl Into<String>) -> Self {
        Self {
            code: Self::INVALID_REQUEST,
            message: "Invalid Request".to_string(),
            data: Some(serde_json::Value::String(details.into())),
        }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: Self::METHOD_NOT_FOUND,
            message: format!("Method not found: {}", method),
            data: None,
        }
    }

    pub fn invalid_params(details: impl Into<String>) -> Self {
        Self {
            code: Self::INVALID_PARAMS,
            message: "Invalid params".to_string(),
            data: Some(serde_json::Value::String(details.into())),
        }
    }

    pub fn internal(details: impl Into<String>) -> Self {
        Self {
            code: Self::INTERNAL_ERROR,
            message: "Internal error".to_string(),
            data: Some(serde_json::Value::String(details.into())),
        }
    }
}

/// JSON-RPC 2.0 Request Object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    #[serde(default)]
    pub id: Option<RpcId>,
}

impl RpcRequest {
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// JSON-RPC 2.0 Response Object
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    pub id: Option<RpcId>,
}

impl RpcResponse {
    pub fn success(id: Option<RpcId>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id: Some(id.unwrap_or(RpcId::Null)),
        }
    }

    pub fn error(id: Option<RpcId>, error: RpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(error),
            id: Some(id.unwrap_or(RpcId::Null)),
        }
    }
}

#[derive(Debug)]
pub enum IncomingMessage {
    Single(serde_json::Value),
    Batch(Vec<serde_json::Value>),
}

/// Parse incoming JSON text into an IncomingMessage (single or batch)
pub fn parse_incoming(raw: &str) -> Result<IncomingMessage, RpcError> {
    let parsed: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| RpcError::parse_error(e.to_string()))?;

    match parsed {
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                Err(RpcError::invalid_request("Batch array cannot be empty"))
            } else {
                Ok(IncomingMessage::Batch(arr))
            }
        }
        serde_json::Value::Object(_) => Ok(IncomingMessage::Single(parsed)),
        _ => Err(RpcError::invalid_request("Expected JSON object or array")),
    }
}

/// Parse a single JSON value into an RpcRequest, returning the id if available on failure
pub fn parse_request(val: serde_json::Value) -> Result<RpcRequest, (Option<RpcId>, RpcError)> {
    let obj = match val.as_object() {
        Some(o) => o,
        None => return Err((None, RpcError::invalid_request("Request must be an object"))),
    };

    let id = obj
        .get("id")
        .and_then(|v| serde_json::from_value::<RpcId>(v.clone()).ok());

    match obj.get("jsonrpc") {
        Some(serde_json::Value::String(v)) if v == "2.0" => {}
        _ => {
            return Err((
                id,
                RpcError::invalid_request("Member 'jsonrpc' must be '2.0'"),
            ));
        }
    }

    let method = match obj.get("method") {
        Some(serde_json::Value::String(m)) => m.clone(),
        _ => {
            return Err((
                id,
                RpcError::invalid_request("Member 'method' must be a string"),
            ));
        }
    };

    let params = obj.get("params").cloned();

    Ok(RpcRequest {
        jsonrpc: "2.0".to_string(),
        method,
        params,
        id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_id_serialization() {
        let id_num = RpcId::Number(42);
        assert_eq!(serde_json::to_string(&id_num).unwrap(), "42");

        let id_str = RpcId::String("req-1".to_string());
        assert_eq!(serde_json::to_string(&id_str).unwrap(), "\"req-1\"");

        let id_null = RpcId::Null;
        assert_eq!(serde_json::to_string(&id_null).unwrap(), "null");
    }

    #[test]
    fn test_parse_request_success() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "sparky.pet",
            "id": 1
        });
        let req = parse_request(json).expect("Expected valid request");
        assert_eq!(req.method, "sparky.pet");
        assert_eq!(req.id, Some(RpcId::Number(1)));
        assert!(!req.is_notification());
    }

    #[test]
    fn test_parse_notification() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "sparky.poke"
        });
        let req = parse_request(json).expect("Expected valid notification");
        assert_eq!(req.method, "sparky.poke");
        assert_eq!(req.id, None);
        assert!(req.is_notification());
    }

    #[test]
    fn test_parse_request_invalid_version() {
        let json = serde_json::json!({
            "jsonrpc": "1.0",
            "method": "sparky.pet",
            "id": 1
        });
        let err = parse_request(json).unwrap_err();
        assert_eq!(err.0, Some(RpcId::Number(1)));
        assert_eq!(err.1.code, RpcError::INVALID_REQUEST);
    }

    #[test]
    fn test_parse_incoming_empty_batch() {
        let res = parse_incoming("[]");
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().code, RpcError::INVALID_REQUEST);
    }
}

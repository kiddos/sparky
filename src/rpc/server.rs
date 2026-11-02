use serde_json::json;
use std::io::{BufRead, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::rpc::handler::{RpcCommand, is_main_thread_method, try_handle_direct};
use crate::rpc::protocol::{IncomingMessage, RpcError, RpcResponse, parse_incoming, parse_request};
use crate::state::EmotionRegistry;

/// Trait to notify the GLFW event loop to wake up immediately
pub trait WakeNotifier: Send + Sync {
    fn wake(&self);
}

/// GLFW wake notifier using ThreadSafeGlfw to post empty events from any thread
pub struct GlfwWakeNotifier {
    glfw: Arc<parking_lot::Mutex<glfw::ThreadSafeGlfw>>,
}

impl GlfwWakeNotifier {
    pub fn new(glfw: glfw::ThreadSafeGlfw) -> Self {
        Self {
            glfw: Arc::new(parking_lot::Mutex::new(glfw)),
        }
    }
}

impl WakeNotifier for GlfwWakeNotifier {
    fn wake(&self) {
        self.glfw.lock().post_empty_event();
    }
}

/// No-op wake notifier for testing or environments without a window
#[allow(dead_code)]
pub struct NoopWakeNotifier;

impl WakeNotifier for NoopWakeNotifier {
    fn wake(&self) {}
}

/// JSON-RPC 2.0 Server supporting both HTTP POST and raw line-delimited TCP
pub struct RpcServer {
    local_addr: SocketAddr,
    cmd_rx: crossbeam_channel::Receiver<RpcCommand>,
    running: Arc<AtomicBool>,
    handle: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl RpcServer {
    /// Starts the RPC server on the specified address (e.g. "127.0.0.1:4040")
    pub fn start(
        addr: &str,
        registry: Arc<EmotionRegistry>,
        wake_notifier: Box<dyn WakeNotifier>,
    ) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let local_addr = listener.local_addr()?;

        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<RpcCommand>();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let wake_notifier: Arc<dyn WakeNotifier> = Arc::from(wake_notifier);

        let handle = std::thread::Builder::new()
            .name("sparky-rpc-listener".to_string())
            .spawn(move || {
                while running_clone.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _peer)) => {
                            if !running_clone.load(Ordering::SeqCst) {
                                break;
                            }
                            let reg = registry.clone();
                            let tx = cmd_tx.clone();
                            let wake = wake_notifier.clone();

                            std::thread::Builder::new()
                                .name("sparky-rpc-conn".to_string())
                                .spawn(move || {
                                    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                                    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));

                                    let mut peek_buf = [0u8; 16];
                                    let n = stream.peek(&mut peek_buf).unwrap_or(0);
                                    let peek_slice = &peek_buf[..n];

                                    let is_http = peek_slice.starts_with(b"POST ")
                                        || peek_slice.starts_with(b"GET ")
                                        || peek_slice.starts_with(b"OPTIONS ")
                                        || peek_slice.starts_with(b"HEAD ")
                                        || peek_slice.starts_with(b"PUT ")
                                        || peek_slice.starts_with(b"DELETE ");

                                    if is_http {
                                        let _ = handle_http_client(stream, &reg, &tx, &wake);
                                    } else {
                                        let _ = handle_raw_tcp_client(stream, &reg, &tx, &wake);
                                    }
                                })
                                .ok();
                        }
                        Err(_) => {
                            if !running_clone.load(Ordering::SeqCst) {
                                break;
                            }
                        }
                    }
                }
            })?;

        Ok(Self {
            local_addr,
            cmd_rx,
            running,
            handle: parking_lot::Mutex::new(Some(handle)),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn try_recv_command(&self) -> Option<RpcCommand> {
        self.cmd_rx.try_recv().ok()
    }

    #[allow(dead_code)]
    pub fn recv_command_timeout(
        &self,
        timeout: Duration,
    ) -> Result<RpcCommand, crossbeam_channel::RecvTimeoutError> {
        self.cmd_rx.recv_timeout(timeout)
    }

    pub fn shutdown(&self) {
        if self.running.swap(false, Ordering::SeqCst) {
            // Unblock listener.accept() by connecting to self
            let _ = TcpStream::connect_timeout(&self.local_addr, Duration::from_millis(200));
            if let Some(h) = self.handle.lock().take() {
                let _ = h.join();
            }
        }
    }
}

impl Drop for RpcServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Handles HTTP clients (curl, browser fetch, REST/RPC tools)
fn handle_http_client(
    mut stream: TcpStream,
    registry: &Arc<EmotionRegistry>,
    cmd_tx: &crossbeam_channel::Sender<RpcCommand>,
    wake_notifier: &Arc<dyn WakeNotifier>,
) -> anyhow::Result<()> {
    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut header_lines = Vec::new();
    let mut line = String::new();

    while reader.read_line(&mut line)? > 0 {
        if line == "\r\n" || line == "\n" {
            break;
        }
        header_lines.push(line.trim().to_string());
        line.clear();
    }

    if header_lines.is_empty() {
        return Ok(());
    }

    let request_line = &header_lines[0];
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");

    if method == "OPTIONS" {
        let resp = "HTTP/1.1 200 OK\r\n\
            Access-Control-Allow-Origin: *\r\n\
            Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n\
            Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
            Content-Length: 0\r\n\
            Connection: close\r\n\
            \r\n";
        stream.write_all(resp.as_bytes())?;
        return Ok(());
    }

    if method == "GET" {
        let body = json!({
            "status": "ok",
            "service": "sparky-json-rpc",
            "version": env!("CARGO_PKG_VERSION"),
            "path": path,
            "info": "Sparky JSON-RPC 2.0 server. Send POST requests with JSON-RPC payloads."
        })
        .to_string();

        let resp = format!(
            "HTTP/1.1 200 OK\r\n\
            Content-Type: application/json\r\n\
            Access-Control-Allow-Origin: *\r\n\
            Content-Length: {}\r\n\
            Connection: close\r\n\
            \r\n\
            {}",
            body.len(),
            body
        );
        stream.write_all(resp.as_bytes())?;
        return Ok(());
    }

    if method != "POST" {
        let resp = "HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n";
        stream.write_all(resp.as_bytes())?;
        return Ok(());
    }

    let mut content_length: usize = 0;
    for h in &header_lines[1..] {
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse::<usize>().unwrap_or(0);
            }
        }
    }

    if content_length == 0 {
        let err_resp = RpcResponse::error(
            None,
            RpcError::invalid_request("Missing or zero Content-Length for POST request"),
        );
        let body = serde_json::to_string(&err_resp)?;
        let resp = format!(
            "HTTP/1.1 400 Bad Request\r\n\
            Content-Type: application/json\r\n\
            Access-Control-Allow-Origin: *\r\n\
            Content-Length: {}\r\n\
            Connection: close\r\n\
            \r\n\
            {}",
            body.len(),
            body
        );
        stream.write_all(resp.as_bytes())?;
        return Ok(());
    }

    let mut body_bytes = vec![0u8; content_length];
    reader.read_exact(&mut body_bytes)?;
    let body_str = String::from_utf8_lossy(&body_bytes);

    let rpc_response = process_json_rpc(&body_str, registry, cmd_tx, wake_notifier);

    match rpc_response {
        Some(response_json) => {
            let resp = format!(
                "HTTP/1.1 200 OK\r\n\
                Content-Type: application/json\r\n\
                Access-Control-Allow-Origin: *\r\n\
                Content-Length: {}\r\n\
                Connection: close\r\n\
                \r\n\
                {}",
                response_json.len(),
                response_json
            );
            stream.write_all(resp.as_bytes())?;
        }
        None => {
            let resp = "HTTP/1.1 204 No Content\r\n\
                Access-Control-Allow-Origin: *\r\n\
                Connection: close\r\n\
                \r\n";
            stream.write_all(resp.as_bytes())?;
        }
    }

    Ok(())
}

/// Handles raw line-delimited TCP socket connections (nc, socat, TCP streams)
fn handle_raw_tcp_client(
    mut stream: TcpStream,
    registry: &Arc<EmotionRegistry>,
    cmd_tx: &crossbeam_channel::Sender<RpcCommand>,
    wake_notifier: &Arc<dyn WakeNotifier>,
) -> anyhow::Result<()> {
    let reader = std::io::BufReader::new(stream.try_clone()?);

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let rpc_response = process_json_rpc(trimmed, registry, cmd_tx, wake_notifier);
        if let Some(resp_json) = rpc_response {
            stream.write_all(resp_json.as_bytes())?;
            stream.write_all(b"\n")?;
            stream.flush()?;
        }
    }

    Ok(())
}

/// Processes a raw JSON-RPC string (single or batch) and returns serialized JSON response if any
pub fn process_json_rpc(
    raw: &str,
    registry: &Arc<EmotionRegistry>,
    cmd_tx: &crossbeam_channel::Sender<RpcCommand>,
    wake_notifier: &Arc<dyn WakeNotifier>,
) -> Option<String> {
    let incoming = match parse_incoming(raw) {
        Ok(inc) => inc,
        Err(err) => {
            let resp = RpcResponse::error(None, err);
            return serde_json::to_string(&resp).ok();
        }
    };

    match incoming {
        IncomingMessage::Single(val) => {
            let resp = process_single_value(val, registry, cmd_tx, wake_notifier);
            resp.and_then(|r| serde_json::to_string(&r).ok())
        }
        IncomingMessage::Batch(items) => {
            let mut responses = Vec::new();
            for item in items {
                if let Some(resp) = process_single_value(item, registry, cmd_tx, wake_notifier) {
                    responses.push(resp);
                }
            }
            if responses.is_empty() {
                None
            } else {
                serde_json::to_string(&responses).ok()
            }
        }
    }
}

/// Processes a single JSON-RPC value, handling notifications and requests
fn process_single_value(
    val: serde_json::Value,
    registry: &Arc<EmotionRegistry>,
    cmd_tx: &crossbeam_channel::Sender<RpcCommand>,
    wake_notifier: &Arc<dyn WakeNotifier>,
) -> Option<RpcResponse> {
    let req = match parse_request(val) {
        Ok(r) => r,
        Err((id, err)) => return Some(RpcResponse::error(id, err)),
    };

    // Direct resolution for queries that don't need main-thread access
    if let Some(direct_result) = try_handle_direct(&req.method, registry) {
        if req.is_notification() {
            return None;
        }
        return Some(match direct_result {
            Ok(v) => RpcResponse::success(req.id, v),
            Err(e) => RpcResponse::error(req.id, e),
        });
    }

    if !is_main_thread_method(&req.method) {
        if req.is_notification() {
            return None;
        }
        return Some(RpcResponse::error(
            req.id,
            RpcError::method_not_found(&req.method),
        ));
    }

    let is_notif = req.is_notification();
    let id = req.id;
    let method = req.method;
    let params = req.params.unwrap_or(serde_json::Value::Null);

    // Forward to main thread
    let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);
    let cmd = RpcCommand {
        method,
        params,
        responder: if is_notif { None } else { Some(resp_tx) },
    };

    if let Err(e) = cmd_tx.send(cmd) {
        if is_notif {
            return None;
        }
        return Some(RpcResponse::error(
            id,
            RpcError::internal(format!("Main thread channel closed: {}", e)),
        ));
    }

    wake_notifier.wake();

    if is_notif {
        return None;
    }

    match resp_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(val)) => Some(RpcResponse::success(id, val)),
        Ok(Err(err)) => Some(RpcResponse::error(id, err)),
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => Some(RpcResponse::error(
            id,
            RpcError::internal("Main thread request timed out (5s)"),
        )),
        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => Some(RpcResponse::error(
            id,
            RpcError::internal("Main thread terminated without responding"),
        )),
    }
}

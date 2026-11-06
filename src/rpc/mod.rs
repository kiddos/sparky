pub mod handler;
pub mod protocol;
pub mod server;

#[cfg(all(test, not(target_os = "windows")))]
pub mod behavior_tests;

#[allow(unused_imports)]
pub use handler::{
    RpcCommand, base64_decode, base64_encode, encode_png_rgb, encode_png_rgba,
    execute_main_thread_command, extract_bool_param, extract_pos_params, extract_string_param,
    handle_rpc_command,
};
#[allow(unused_imports)]
pub use protocol::{RpcError, RpcId, RpcRequest, RpcResponse};
#[allow(unused_imports)]
pub use server::{GlfwWakeNotifier, NoopWakeNotifier, RpcServer, WakeNotifier};

#[cfg(test)]
mod tests {
    use crate::rpc::server::{NoopWakeNotifier, RpcServer};
    use crate::state::EmotionRegistry;
    use serde_json::json;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_rpc_server_direct_methods_over_tcp() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions"),
        );

        let server = RpcServer::start("127.0.0.1:0", registry, Box::new(NoopWakeNotifier))
            .expect("Failed to bind RPC server");
        let addr = server.local_addr();

        // Connect over raw TCP
        use std::io::{BufRead, BufReader, Write};
        let mut stream = std::net::TcpStream::connect(addr).expect("Failed to connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();

        // 1. sparky.version
        let req = json!({
            "jsonrpc": "2.0",
            "method": "sparky.version",
            "id": 1
        });
        writeln!(stream, "{}", req).expect("Failed to write request");

        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut response_line = String::new();
        reader
            .read_line(&mut response_line)
            .expect("Failed to read line");

        let resp: serde_json::Value =
            serde_json::from_str(response_line.trim()).expect("Failed to parse response");
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["name"], "sparky");

        // 2. sparky.list_emotions
        let req2 = json!({
            "jsonrpc": "2.0",
            "method": "sparky.list_emotions",
            "id": 2
        });
        writeln!(stream, "{}", req2).expect("Failed to write request");
        response_line.clear();
        reader
            .read_line(&mut response_line)
            .expect("Failed to read line");

        let resp2: serde_json::Value =
            serde_json::from_str(response_line.trim()).expect("Failed to parse response");
        assert_eq!(resp2["id"], 2);
        assert!(resp2["result"].as_array().unwrap().len() >= 4);

        // 3. rpc.discover
        let req3 = json!({
            "jsonrpc": "2.0",
            "method": "rpc.discover",
            "id": 3
        });
        writeln!(stream, "{}", req3).expect("Failed to write request");
        response_line.clear();
        reader
            .read_line(&mut response_line)
            .expect("Failed to read line");

        let resp3: serde_json::Value =
            serde_json::from_str(response_line.trim()).expect("Failed to parse response");
        assert_eq!(resp3["id"], 3);
        assert_eq!(resp3["result"]["openrpc"], "1.3.2");

        server.shutdown();
    }

    #[test]
    fn test_rpc_server_over_http() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions"),
        );

        let server = RpcServer::start("127.0.0.1:0", registry, Box::new(NoopWakeNotifier))
            .expect("Failed to bind RPC server");
        let addr = server.local_addr();

        use std::io::{Read, Write};
        let mut stream = std::net::TcpStream::connect(addr).expect("Failed to connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();

        let req_body = json!({
            "jsonrpc": "2.0",
            "method": "sparky.version",
            "id": "http-1"
        })
        .to_string();

        let http_request = format!(
            "POST / HTTP/1.1\r\n\
            Host: {}\r\n\
            Content-Type: application/json\r\n\
            Content-Length: {}\r\n\
            Connection: close\r\n\
            \r\n\
            {}",
            addr,
            req_body.len(),
            req_body
        );

        stream
            .write_all(http_request.as_bytes())
            .expect("Failed to write HTTP request");

        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("Failed to read response");

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("\"id\":\"http-1\""));
        assert!(response.contains("\"name\":\"sparky\""));

        server.shutdown();
    }

    #[test]
    fn test_rpc_server_dispatches_to_main_thread_receiver() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions"),
        );

        let server = RpcServer::start("127.0.0.1:0", registry, Box::new(NoopWakeNotifier))
            .expect("Failed to bind RPC server");
        let addr = server.local_addr();

        // Spawn mock client in background thread
        let client_thread = std::thread::spawn(move || {
            use std::io::{BufRead, BufReader, Write};
            let mut stream = std::net::TcpStream::connect(addr).expect("Client connect failed");
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();

            let req = json!({
                "jsonrpc": "2.0",
                "method": "sparky.set_emotion",
                "params": {"emotion": "happy"},
                "id": 99
            });
            writeln!(stream, "{}", req).expect("Write failed");

            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("Read failed");
            let resp: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
            resp
        });

        // Simulate main loop receiving command
        let cmd = server
            .recv_command_timeout(Duration::from_secs(2))
            .expect("Expected command on server");
        assert_eq!(cmd.method, "sparky.set_emotion");
        assert_eq!(cmd.params["emotion"], "happy");

        // Simulate main loop responding
        if let Some(responder) = cmd.responder {
            responder
                .send(Ok(json!({
                    "success": true,
                    "emotion": "happy",
                    "active_clip": "happy"
                })))
                .unwrap();
        }

        let client_resp = client_thread.join().unwrap();
        assert_eq!(client_resp["id"], 99);
        assert_eq!(client_resp["result"]["success"], true);
        assert_eq!(client_resp["result"]["emotion"], "happy");

        server.shutdown();
    }

    #[test]
    fn test_rpc_batch_requests() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions"),
        );

        let server = RpcServer::start("127.0.0.1:0", registry, Box::new(NoopWakeNotifier))
            .expect("Failed to bind RPC server");
        let addr = server.local_addr();

        use std::io::{BufRead, BufReader, Write};
        let mut stream = std::net::TcpStream::connect(addr).expect("Connect failed");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();

        // Send batch: version (request) + list_emotions (request) + notification
        let batch = json!([
            {"jsonrpc": "2.0", "method": "sparky.version", "id": 10},
            {"jsonrpc": "2.0", "method": "sparky.version"}, // notification
            {"jsonrpc": "2.0", "method": "non_existent_method", "id": 11}
        ]);
        writeln!(stream, "{}", batch).expect("Write failed");

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).expect("Read failed");

        let responses: Vec<serde_json::Value> = serde_json::from_str(line.trim()).unwrap();
        // Notification produces no response, so only 2 responses in batch
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["id"], 10);
        assert_eq!(responses[0]["result"]["name"], "sparky");
        assert_eq!(responses[1]["id"], 11);
        assert_eq!(responses[1]["error"]["code"], -32601); // method not found

        server.shutdown();
    }

    #[test]
    fn test_rpc_http_cors_options() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions"),
        );

        let server = RpcServer::start("127.0.0.1:0", registry, Box::new(NoopWakeNotifier))
            .expect("Failed to bind RPC server");
        let addr = server.local_addr();

        use std::io::{Read, Write};
        let mut stream = std::net::TcpStream::connect(addr).expect("Connect failed");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();

        let req = format!(
            "OPTIONS / HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            addr
        );
        stream.write_all(req.as_bytes()).unwrap();

        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("Access-Control-Allow-Origin: *"));
        assert!(response.contains("Access-Control-Allow-Methods: POST, GET, OPTIONS"));

        server.shutdown();
    }

    #[test]
    fn test_rpc_get_frame_dispatch() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions"),
        );

        let server = RpcServer::start("127.0.0.1:0", registry, Box::new(NoopWakeNotifier))
            .expect("Failed to bind RPC server");
        let addr = server.local_addr();

        let client_thread = std::thread::spawn(move || {
            use std::io::{BufRead, BufReader, Write};
            let mut stream = std::net::TcpStream::connect(addr).expect("Client connect failed");
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();

            let req = json!({
                "jsonrpc": "2.0",
                "method": "sparky.get_frame",
                "params": {"source": "clip", "include_data": false},
                "id": 55
            });
            writeln!(stream, "{}", req).expect("Write failed");

            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("Read failed");
            let resp: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
            resp
        });

        let cmd = server
            .recv_command_timeout(Duration::from_secs(2))
            .expect("Expected command on server");
        assert_eq!(cmd.method, "sparky.get_frame");
        assert_eq!(cmd.params["source"], "clip");
        assert_eq!(cmd.params["include_data"], false);

        if let Some(responder) = cmd.responder {
            responder
                .send(Ok(json!({
                    "emotion": "sit",
                    "clip": "sit",
                    "frame_index": 0,
                    "total_frames": 192,
                    "width": 384,
                    "height": 216,
                    "channels": 3,
                    "source": "clip",
                    "format": "image/png"
                })))
                .unwrap();
        }

        let client_resp = client_thread.join().unwrap();
        assert_eq!(client_resp["id"], 55);
        assert_eq!(client_resp["result"]["format"], "image/png");
        assert_eq!(client_resp["result"]["source"], "clip");

        server.shutdown();
    }
}

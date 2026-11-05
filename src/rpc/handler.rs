use crate::animation::{AnimationClip, FramePacer};
use crate::brain::BrainEngine;
use crate::render::PetWindow;
use crate::rpc::protocol::RpcError;
use crate::state::{EmotionRegistry, StateMachine};
use crate::tray::TrayService;
use crate::ui::ChatState;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

/// Internal command dispatched to the main thread
pub struct RpcCommand {
    pub method: String,
    pub params: serde_json::Value,
    pub responder: Option<crossbeam_channel::Sender<Result<serde_json::Value, RpcError>>>,
}

/// Standard RFC 4648 Base64 encoder without external dependencies
pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks_exact(3);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
        out.push(TABLE[(n & 0x3F) as usize] as char);
    }

    match remainder.len() {
        1 => {
            let n = (remainder[0] as u32) << 16;
            out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = ((remainder[0] as u32) << 16) | ((remainder[1] as u32) << 8);
            out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        _ => {}
    }

    out
}

/// Standard RFC 4648 Base64 decoder without external dependencies
#[allow(dead_code)]
pub fn base64_decode(input: &str) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for &b in input.as_bytes() {
        if b == b'=' || b.is_ascii_whitespace() {
            continue;
        }
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => anyhow::bail!("Invalid base64 character: {}", b as char),
        } as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}

/// Encodes raw RGBA pixel buffer to standard PNG bytes
pub fn encode_png_rgba(width: u32, height: u32, rgba_data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let encoder = PngEncoder::new(&mut buf);
    encoder.write_image(rgba_data, width, height, ExtendedColorType::Rgba8)?;
    Ok(buf)
}

/// Encodes raw RGB pixel buffer to standard PNG bytes
pub fn encode_png_rgb(width: u32, height: u32, rgb_data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let encoder = PngEncoder::new(&mut buf);
    encoder.write_image(rgb_data, width, height, ExtendedColorType::Rgb8)?;
    Ok(buf)
}

/// Helper to extract a string parameter from either named fields, positional array, or a bare string
pub fn extract_string_param(params: &serde_json::Value, keys: &[&str]) -> Option<String> {
    match params {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => {
            arr.first().and_then(|v| v.as_str()).map(|s| s.to_string())
        }
        serde_json::Value::Object(map) => {
            for key in keys {
                if let Some(val) = map.get(*key).and_then(|v| v.as_str()) {
                    return Some(val.to_string());
                }
            }
            None
        }
        _ => None,
    }
}

/// Helper to extract a boolean parameter from named fields, positional array, or a bare bool
pub fn extract_bool_param(params: &serde_json::Value, keys: &[&str]) -> Option<bool> {
    match params {
        serde_json::Value::Bool(b) => Some(*b),
        serde_json::Value::Array(arr) => arr.first().and_then(|v| v.as_bool()),
        serde_json::Value::Object(map) => {
            for key in keys {
                if let Some(val) = map.get(*key).and_then(|v| v.as_bool()) {
                    return Some(val);
                }
            }
            None
        }
        _ => None,
    }
}

/// Helper to extract (x, y) coordinates from named fields or positional array
pub fn extract_pos_params(params: &serde_json::Value) -> Option<(i32, i32)> {
    match params {
        serde_json::Value::Array(arr) => {
            if arr.len() >= 2 {
                let x = arr[0].as_i64()? as i32;
                let y = arr[1].as_i64()? as i32;
                Some((x, y))
            } else {
                None
            }
        }
        serde_json::Value::Object(map) => {
            let x = map.get("x").and_then(|v| v.as_i64())? as i32;
            let y = map.get("y").and_then(|v| v.as_i64())? as i32;
            Some((x, y))
        }
        _ => None,
    }
}

/// Returns true if the method must be executed on the main thread
pub fn is_main_thread_method(method: &str) -> bool {
    matches!(
        method,
        "sparky.get_state"
            | "sparky.get_frame"
            | "sparky.set_emotion"
            | "sparky.complete_animation"
            | "sparky.clear_queue"
            | "sparky.pet"
            | "sparky.poke"
            | "sparky.touch"
            | "sparky.send_chat"
            | "sparky.get_chat_history"
            | "sparky.clear_chat"
            | "sparky.set_visible"
            | "sparky.set_click_through"
            | "sparky.set_chat_visible"
            | "sparky.set_position"
            | "sparky.get_position"
            | "sparky.quit"
    )
}

/// Checks if an RPC method can be handled directly by the worker thread without main-thread synchronization
pub fn try_handle_direct(
    method: &str,
    registry: &Arc<EmotionRegistry>,
) -> Option<Result<serde_json::Value, RpcError>> {
    match method {
        "sparky.list_emotions" => {
            let emotions: Vec<_> = registry.all_emotions().collect();
            Some(serde_json::to_value(&emotions).map_err(|e| RpcError::internal(e.to_string())))
        }
        "sparky.version" => Some(Ok(json!({
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
        }))),
        "rpc.discover" => Some(Ok(discover_rpc_methods())),
        _ => None,
    }
}

/// Discovers and documents all RPC methods supported by Sparky
pub fn discover_rpc_methods() -> serde_json::Value {
    json!({
        "openrpc": "1.3.2",
        "info": {
            "title": "Sparky Desktop Pet JSON-RPC API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "JSON-RPC 2.0 interface for controlling Sparky desktop pet visuals, emotions, AI chat, and window geometry"
        },
        "methods": [
            {
                "name": "sparky.get_state",
                "description": "Returns full desktop pet state including current emotion, active clip, window position and visibility, and chat status",
                "params": [],
                "result": { "description": "Current state object" }
            },
            {
                "name": "sparky.get_frame",
                "description": "Captures the current display frame (from rendered window framebuffer or animation clip) as PNG with metadata and base64 encoding",
                "params": [
                    { "name": "source", "description": "'framebuffer' (rendered window) or 'clip' (source animation frame)", "required": false, "schema": { "type": "string" } },
                    { "name": "include_data", "description": "Whether to include base64 PNG data in the response (default true)", "required": false, "schema": { "type": "boolean" } },
                    { "name": "save_path", "description": "Optional file path to write PNG image directly to disk", "required": false, "schema": { "type": "string" } }
                ],
                "result": { "description": "Frame metadata, dimensions, clip/emotion status, and base64 PNG" }
            },
            {
                "name": "sparky.set_emotion",
                "description": "Pushes a target emotion (e.g. sit, happy, angry, naughty, sad, smirk, proud) into the emotion queue to play after current animation completes",
                "params": [
                    { "name": "emotion", "description": "Target emotion ID to push to queue", "required": true, "schema": { "type": "string" } }
                ],
                "result": { "description": "Queue confirmation with resolved clip, tint, and updated queue" }
            },
            {
                "name": "sparky.complete_animation",
                "description": "Simulates completion of the current animation cycle, consuming the next emotion from the queue or keeping current emotion if empty",
                "params": [],
                "result": { "description": "Updated emotion state after consuming queue" }
            },
            {
                "name": "sparky.clear_queue",
                "description": "Clears all pending emotions in the emotion queue",
                "params": [],
                "result": { "description": "Confirmation of cleared queue" }
            },
            {
                "name": "sparky.list_emotions",
                "description": "Lists all registered emotions and their configurations (loop mode, duration, fallback, tint)",
                "params": [],
                "result": { "description": "Array of emotion definitions" }
            },
            {
                "name": "sparky.pet",
                "description": "Simulates petting Sparky, transitioning him to 'happy'",
                "params": [],
                "result": { "description": "Result indicating transition to happy" }
            },
            {
                "name": "sparky.poke",
                "description": "Simulates poking Sparky, transitioning him to 'angry'",
                "params": [],
                "result": { "description": "Result indicating transition to angry" }
            },
            {
                "name": "sparky.touch",
                "description": "Registers user interaction, resetting the boredom timer",
                "params": [],
                "result": { "description": "Success confirmation" }
            },
            {
                "name": "sparky.send_chat",
                "description": "Sends a message/prompt to Sparky's cognitive brain engine",
                "params": [
                    { "name": "prompt", "description": "Message text to send to Sparky", "required": true, "schema": { "type": "string" } }
                ],
                "result": { "description": "Confirmation of queued prompt" }
            },
            {
                "name": "sparky.get_chat_history",
                "description": "Returns the conversation message history",
                "params": [],
                "result": { "description": "Array of {sender, message} objects" }
            },
            {
                "name": "sparky.clear_chat",
                "description": "Clears conversation history and resets to default greeting",
                "params": [],
                "result": { "description": "Success confirmation" }
            },
            {
                "name": "sparky.set_visible",
                "description": "Show or hide the desktop pet window",
                "params": [
                    { "name": "visible", "description": "Window visibility state", "required": true, "schema": { "type": "boolean" } }
                ],
                "result": { "description": "Updated visibility state" }
            },
            {
                "name": "sparky.set_click_through",
                "description": "Enable or disable mouse click-through mode",
                "params": [
                    { "name": "click_through", "description": "Click-through state", "required": true, "schema": { "type": "boolean" } }
                ],
                "result": { "description": "Updated click-through state" }
            },
            {
                "name": "sparky.set_chat_visible",
                "description": "Toggle egui chat overlay visibility",
                "params": [
                    { "name": "visible", "description": "Chat visibility state", "required": true, "schema": { "type": "boolean" } }
                ],
                "result": { "description": "Updated chat visibility state" }
            },
            {
                "name": "sparky.set_position",
                "description": "Sets the desktop position (x, y) of the pet window",
                "params": [
                    { "name": "x", "description": "Screen X position", "required": true, "schema": { "type": "integer" } },
                    { "name": "y", "description": "Screen Y position", "required": true, "schema": { "type": "integer" } }
                ],
                "result": { "description": "Updated window coordinates" }
            },
            {
                "name": "sparky.get_position",
                "description": "Gets current screen position (x, y) of the pet window",
                "params": [],
                "result": { "description": "Object with {x, y}" }
            },
            {
                "name": "sparky.quit",
                "description": "Gracefully terminates Sparky application",
                "params": [],
                "result": { "description": "Shutdown status" }
            },
            {
                "name": "sparky.version",
                "description": "Returns application version and metadata",
                "params": [],
                "result": { "description": "Version metadata" }
            },
            {
                "name": "rpc.discover",
                "description": "Discovers all supported RPC methods and specifications",
                "params": [],
                "result": { "description": "OpenRPC discovery schema" }
            }
        ]
    })
}

/// Executes an RPC command on the main thread
#[allow(clippy::too_many_arguments)]
pub fn execute_main_thread_command(
    method: &str,
    params: &serde_json::Value,
    state_machine: &mut StateMachine,
    emotion_registry: &EmotionRegistry,
    pet_window: &mut PetWindow,
    chat_state: &mut ChatState,
    brain: &BrainEngine,
    tray_service: &Option<TrayService>,
    frame_pacer: &mut FramePacer,
    is_visible: &mut bool,
    clips: &HashMap<String, AnimationClip>,
) -> Result<serde_json::Value, RpcError> {
    match method {
        "sparky.get_state" => {
            let (win_x, win_y) = pet_window.window.get_pos();
            let (win_w, win_h) = pet_window.window.get_size();
            Ok(json!({
                "emotion": state_machine.current_emotion,
                "queue": state_machine.emotion_queue.iter().collect::<Vec<_>>(),
                "emotion_queue": state_machine.emotion_queue.iter().collect::<Vec<_>>(),
                "active_clip": state_machine.active_clip_name,
                "active_tint": state_machine.active_tint,
                "window": {
                    "x": win_x,
                    "y": win_y,
                    "width": win_w,
                    "height": win_h,
                    "visible": *is_visible,
                    "click_through": pet_window.is_click_through,
                },
                "chat": {
                    "visible": chat_state.is_visible,
                    "message_count": chat_state.messages.len(),
                    "is_generating": chat_state.is_generating,
                },
                "fps": frame_pacer.target_fps(),
            }))
        }
        "sparky.get_frame" => {
            let requested_source = extract_string_param(params, &["source"])
                .unwrap_or_else(|| "framebuffer".to_string());
            let include_data = extract_bool_param(params, &["include_data"]).unwrap_or(true);
            let save_path = extract_string_param(params, &["save_path", "path", "file"]);

            let active_clip_name = &state_machine.active_clip_name;
            let frame_idx = frame_pacer.current_index();
            let total_frames = clips
                .get(active_clip_name)
                .map(|c| c.frames.len())
                .unwrap_or(0);

            let (width, height, channels, png_bytes, actual_source) =
                if requested_source == "clip" || !*is_visible {
                    if let Some(clip) = clips.get(active_clip_name) {
                        if let Some(frame) = clip.frames.get(frame_idx) {
                            let png = encode_png_rgb(frame.width, frame.height, &frame.rgb_data)
                                .map_err(|e| {
                                    RpcError::internal(format!("PNG encoding error: {}", e))
                                })?;
                            (frame.width, frame.height, 3, png, "clip".to_string())
                        } else {
                            return Err(RpcError::internal(format!(
                                "Frame index {} out of bounds for clip '{}' (total frames: {})",
                                frame_idx,
                                active_clip_name,
                                clip.frames.len()
                            )));
                        }
                    } else {
                        return Err(RpcError::internal(format!(
                            "Active clip '{}' not found in loaded clips",
                            active_clip_name
                        )));
                    }
                } else {
                    let (fb_w, fb_h) = pet_window.window.get_framebuffer_size();
                    if fb_w <= 0 || fb_h <= 0 {
                        return Err(RpcError::internal("Invalid framebuffer dimensions"));
                    }
                    let w = fb_w as u32;
                    let h = fb_h as u32;
                    let mut pixels = vec![0u8; (w * h * 4) as usize];
                    unsafe {
                        gl::PixelStorei(gl::PACK_ALIGNMENT, 1);
                        gl::ReadPixels(
                            0,
                            0,
                            fb_w,
                            fb_h,
                            gl::RGBA,
                            gl::UNSIGNED_BYTE,
                            pixels.as_mut_ptr() as *mut _,
                        );
                    }
                    let stride = (w * 4) as usize;
                    let mut flipped = vec![0u8; pixels.len()];
                    for y in 0..h as usize {
                        let src_row = (h as usize - 1 - y) * stride;
                        let dst_row = y * stride;
                        flipped[dst_row..dst_row + stride]
                            .copy_from_slice(&pixels[src_row..src_row + stride]);
                    }
                    let png = encode_png_rgba(w, h, &flipped)
                        .map_err(|e| RpcError::internal(format!("PNG encoding error: {}", e)))?;
                    (w, h, 4, png, "framebuffer".to_string())
                };

            let mut result = json!({
                "emotion": state_machine.current_emotion,
                "clip": active_clip_name,
                "frame_index": frame_idx,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "channels": channels,
                "source": actual_source,
                "format": "image/png",
                "byte_size": png_bytes.len(),
            });

            if let Some(ref path) = save_path {
                std::fs::write(path, &png_bytes).map_err(|e| {
                    RpcError::internal(format!("Failed to save frame to '{}': {}", path, e))
                })?;
                result["saved_to"] = json!(path);
            }

            if include_data {
                let b64 = base64_encode(&png_bytes);
                result["data_url"] = json!(format!("data:image/png;base64,{}", b64));
                result["base64"] = json!(b64);
            }

            Ok(result)
        }
        "sparky.set_emotion" => {
            let emotion_id =
                extract_string_param(params, &["emotion", "name", "id"]).ok_or_else(|| {
                    RpcError::invalid_params("Missing or invalid 'emotion' parameter")
                })?;

            if emotion_registry.get(&emotion_id).is_some() {
                let (clip, tint) = emotion_registry.resolve_visuals(&emotion_id);
                state_machine.push_emotion(&emotion_id, emotion_registry);
                Ok(json!({
                    "success": true,
                    "emotion": emotion_id,
                    "queued": true,
                    "queue": state_machine.emotion_queue.iter().collect::<Vec<_>>(),
                    "current_emotion": state_machine.current_emotion,
                    "active_clip": clip,
                    "active_tint": tint,
                }))
            } else {
                Err(RpcError::invalid_params(format!(
                    "Unknown emotion '{}'. Available: {:?}",
                    emotion_id,
                    emotion_registry.available_emotion_ids()
                )))
            }
        }
        "sparky.complete_animation" => {
            let transitioned = state_machine.on_animation_completed(emotion_registry);
            if transitioned {
                chat_state.current_emotion_id = state_machine.current_emotion.clone();
                frame_pacer.reset();
            }
            Ok(json!({
                "success": true,
                "transitioned": transitioned,
                "emotion": state_machine.current_emotion,
                "active_clip": state_machine.active_clip_name,
                "active_tint": state_machine.active_tint,
                "queue": state_machine.emotion_queue.iter().collect::<Vec<_>>(),
            }))
        }
        "sparky.clear_queue" => {
            state_machine.clear_queue();
            Ok(json!({
                "success": true,
                "queue": [],
            }))
        }
        "sparky.pet" => {
            state_machine.on_pet(emotion_registry);
            let (clip, tint) = emotion_registry.resolve_visuals("happy");
            Ok(json!({
                "success": true,
                "emotion": "happy",
                "queued": true,
                "queue": state_machine.emotion_queue.iter().collect::<Vec<_>>(),
                "current_emotion": state_machine.current_emotion,
                "active_clip": clip,
                "active_tint": tint,
            }))
        }
        "sparky.poke" => {
            state_machine.on_poke(emotion_registry);
            let (clip, tint) = emotion_registry.resolve_visuals("angry");
            Ok(json!({
                "success": true,
                "emotion": "angry",
                "queued": true,
                "queue": state_machine.emotion_queue.iter().collect::<Vec<_>>(),
                "current_emotion": state_machine.current_emotion,
                "active_clip": clip,
                "active_tint": tint,
            }))
        }
        "sparky.touch" => {
            state_machine.touch_interaction();
            Ok(json!({ "success": true }))
        }
        "sparky.send_chat" => {
            let prompt =
                extract_string_param(params, &["prompt", "message", "text"]).ok_or_else(|| {
                    RpcError::invalid_params("Missing or invalid 'prompt' or 'message' parameter")
                })?;

            chat_state
                .messages
                .push(("User".to_string(), prompt.clone()));
            chat_state
                .messages
                .push(("Sparky".to_string(), String::new()));
            chat_state.is_generating = true;
            brain.send_chat(&prompt);

            Ok(json!({
                "success": true,
                "prompt": prompt,
            }))
        }
        "sparky.get_chat_history" => {
            let history: Vec<_> = chat_state
                .messages
                .iter()
                .map(|(sender, msg)| json!({ "sender": sender, "message": msg }))
                .collect();
            Ok(json!(history))
        }
        "sparky.clear_chat" => {
            chat_state.messages.clear();
            chat_state.messages.push((
                "Sparky".to_string(),
                "Meow! I'm Sparky. How are you today?".to_string(),
            ));
            Ok(json!({ "success": true }))
        }
        "sparky.set_visible" => {
            let visible =
                extract_bool_param(params, &["visible", "value", "enabled"]).ok_or_else(|| {
                    RpcError::invalid_params("Missing or invalid 'visible' boolean parameter")
                })?;

            *is_visible = visible;
            if visible {
                pet_window.window.show();
            } else {
                pet_window.window.hide();
            }

            if let Some(tray) = tray_service {
                tray.show_hide_item.set_checked(visible);
            }

            Ok(json!({ "success": true, "visible": visible }))
        }
        "sparky.set_click_through" => {
            let click_through = extract_bool_param(params, &["click_through", "value", "enabled"])
                .ok_or_else(|| {
                    RpcError::invalid_params("Missing or invalid 'click_through' boolean parameter")
                })?;

            pet_window.set_click_through(click_through);

            if let Some(tray) = tray_service {
                tray.click_through_item.set_checked(click_through);
            }

            Ok(json!({ "success": true, "click_through": click_through }))
        }
        "sparky.set_chat_visible" => {
            let visible =
                extract_bool_param(params, &["visible", "value", "enabled"]).ok_or_else(|| {
                    RpcError::invalid_params("Missing or invalid 'visible' boolean parameter")
                })?;

            chat_state.is_visible = visible;
            if visible {
                chat_state.focus_input = true;
            }
            Ok(json!({ "success": true, "chat_visible": visible }))
        }
        "sparky.set_position" => {
            let (x, y) = extract_pos_params(params).ok_or_else(|| {
                RpcError::invalid_params("Missing or invalid coordinates. Provide {x, y} or [x, y]")
            })?;

            pet_window.set_pos(x, y);
            Ok(json!({ "success": true, "x": x, "y": y }))
        }
        "sparky.get_position" => {
            let (x, y) = pet_window.get_pos();
            Ok(json!({ "x": x, "y": y }))
        }
        "sparky.quit" => {
            pet_window.window.set_should_close(true);
            Ok(json!({ "success": true, "message": "Sparky shutting down" }))
        }
        _ => Err(RpcError::method_not_found(method)),
    }
}

/// Convenience entrypoint for the main event loop to dispatch and reply to an RPC command
#[allow(clippy::too_many_arguments)]
pub fn handle_rpc_command(
    cmd: RpcCommand,
    state_machine: &mut StateMachine,
    emotion_registry: &EmotionRegistry,
    pet_window: &mut PetWindow,
    chat_state: &mut ChatState,
    brain: &BrainEngine,
    tray_service: &Option<TrayService>,
    frame_pacer: &mut FramePacer,
    is_visible: &mut bool,
    clips: &HashMap<String, AnimationClip>,
) {
    let result = execute_main_thread_command(
        &cmd.method,
        &cmd.params,
        state_machine,
        emotion_registry,
        pet_window,
        chat_state,
        brain,
        tray_service,
        frame_pacer,
        is_visible,
        clips,
    );

    if let Some(responder) = cmd.responder {
        let _ = responder.send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_string_param() {
        // String primitive
        let p1 = json!("happy");
        assert_eq!(
            extract_string_param(&p1, &["emotion"]).as_deref(),
            Some("happy")
        );

        // Array
        let p2 = json!(["sit"]);
        assert_eq!(
            extract_string_param(&p2, &["emotion"]).as_deref(),
            Some("sit")
        );

        // Object with primary key
        let p3 = json!({"emotion": "angry"});
        assert_eq!(
            extract_string_param(&p3, &["emotion", "name"]).as_deref(),
            Some("angry")
        );

        // Object with secondary key
        let p4 = json!({"name": "naughty"});
        assert_eq!(
            extract_string_param(&p4, &["emotion", "name"]).as_deref(),
            Some("naughty")
        );

        // Missing
        let p5 = json!({});
        assert_eq!(extract_string_param(&p5, &["emotion"]), None);
    }

    #[test]
    fn test_extract_bool_param() {
        let p1 = json!(true);
        assert_eq!(extract_bool_param(&p1, &["visible"]), Some(true));

        let p2 = json!([false]);
        assert_eq!(extract_bool_param(&p2, &["visible"]), Some(false));

        let p3 = json!({"visible": true});
        assert_eq!(extract_bool_param(&p3, &["visible"]), Some(true));
    }

    #[test]
    fn test_extract_pos_params() {
        let p1 = json!([100, 250]);
        assert_eq!(extract_pos_params(&p1), Some((100, 250)));

        let p2 = json!({"x": 50, "y": 75});
        assert_eq!(extract_pos_params(&p2), Some((50, 75)));

        let p3 = json!([100]);
        assert_eq!(extract_pos_params(&p3), None);
    }

    #[test]
    fn test_discover_rpc_methods() {
        let discovery = discover_rpc_methods();
        assert_eq!(discovery["openrpc"], "1.3.2");
        let methods = discovery["methods"].as_array().expect("Methods array");
        assert!(methods.iter().any(|m| m["name"] == "sparky.get_state"));
        assert!(methods.iter().any(|m| m["name"] == "sparky.get_frame"));
        assert!(methods.iter().any(|m| m["name"] == "sparky.set_emotion"));
        assert!(methods.iter().any(|m| m["name"] == "sparky.send_chat"));
        assert!(methods.iter().any(|m| m["name"] == "sparky.pet"));
        assert!(methods.iter().all(|m| m.get("aliases").is_none()));
    }

    #[test]
    fn test_aliases_removed() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions"),
        );

        // Direct methods reject aliases
        assert!(try_handle_direct("version", &registry).is_none());
        assert!(try_handle_direct("sparky_version", &registry).is_none());
        assert!(try_handle_direct("list_emotions", &registry).is_none());
        assert!(try_handle_direct("get_emotions", &registry).is_none());
        assert!(try_handle_direct("sparky.get_emotions", &registry).is_none());
        assert!(try_handle_direct("sparky_listEmotions", &registry).is_none());
        assert!(try_handle_direct("list_methods", &registry).is_none());
        assert!(try_handle_direct("sparky.list_methods", &registry).is_none());
        assert!(try_handle_direct("rpc_discover", &registry).is_none());

        // Direct methods accept canonical names
        assert!(try_handle_direct("sparky.version", &registry).is_some());
        assert!(try_handle_direct("sparky.list_emotions", &registry).is_some());
        assert!(try_handle_direct("rpc.discover", &registry).is_some());

        // Main thread methods reject aliases
        assert!(!is_main_thread_method("get_state"));
        assert!(!is_main_thread_method("get_status"));
        assert!(!is_main_thread_method("sparky.get_status"));
        assert!(!is_main_thread_method("sparky_getStatus"));
        assert!(!is_main_thread_method("sparky_getState"));
        assert!(!is_main_thread_method("get_frame"));
        assert!(!is_main_thread_method("get_display_frame"));
        assert!(!is_main_thread_method("sparky.get_display_frame"));
        assert!(!is_main_thread_method("sparky_getFrame"));
        assert!(!is_main_thread_method("sparky_getDisplayFrame"));
        assert!(!is_main_thread_method("set_emotion"));
        assert!(!is_main_thread_method("sparky_setEmotion"));
        assert!(!is_main_thread_method("complete_animation"));
        assert!(!is_main_thread_method("sparky_completeAnimation"));
        assert!(!is_main_thread_method("clear_queue"));
        assert!(!is_main_thread_method("sparky_clearQueue"));
        assert!(!is_main_thread_method("pet"));
        assert!(!is_main_thread_method("sparky_pet"));
        assert!(!is_main_thread_method("poke"));
        assert!(!is_main_thread_method("sparky_poke"));
        assert!(!is_main_thread_method("touch"));
        assert!(!is_main_thread_method("sparky_touch"));
        assert!(!is_main_thread_method("send_chat"));
        assert!(!is_main_thread_method("chat"));
        assert!(!is_main_thread_method("sparky.chat"));
        assert!(!is_main_thread_method("sparky_sendChat"));
        assert!(!is_main_thread_method("get_chat_history"));
        assert!(!is_main_thread_method("get_messages"));
        assert!(!is_main_thread_method("sparky_getChatHistory"));
        assert!(!is_main_thread_method("clear_chat"));
        assert!(!is_main_thread_method("sparky_clearChat"));
        assert!(!is_main_thread_method("set_visible"));
        assert!(!is_main_thread_method("sparky_setVisible"));
        assert!(!is_main_thread_method("set_click_through"));
        assert!(!is_main_thread_method("sparky_setClickThrough"));
        assert!(!is_main_thread_method("set_chat_visible"));
        assert!(!is_main_thread_method("sparky_setChatVisible"));
        assert!(!is_main_thread_method("set_position"));
        assert!(!is_main_thread_method("sparky_setPosition"));
        assert!(!is_main_thread_method("get_position"));
        assert!(!is_main_thread_method("sparky_getPosition"));
        assert!(!is_main_thread_method("quit"));
        assert!(!is_main_thread_method("exit"));
        assert!(!is_main_thread_method("sparky_quit"));

        // Main thread methods accept canonical names
        assert!(is_main_thread_method("sparky.get_state"));
        assert!(is_main_thread_method("sparky.get_frame"));
        assert!(is_main_thread_method("sparky.set_emotion"));
        assert!(is_main_thread_method("sparky.complete_animation"));
        assert!(is_main_thread_method("sparky.clear_queue"));
        assert!(is_main_thread_method("sparky.pet"));
        assert!(is_main_thread_method("sparky.poke"));
        assert!(is_main_thread_method("sparky.touch"));
        assert!(is_main_thread_method("sparky.send_chat"));
        assert!(is_main_thread_method("sparky.get_chat_history"));
        assert!(is_main_thread_method("sparky.clear_chat"));
        assert!(is_main_thread_method("sparky.set_visible"));
        assert!(is_main_thread_method("sparky.set_click_through"));
        assert!(is_main_thread_method("sparky.set_chat_visible"));
        assert!(is_main_thread_method("sparky.set_position"));
        assert!(is_main_thread_method("sparky.get_position"));
        assert!(is_main_thread_method("sparky.quit"));
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");

        // Decode roundtrip
        for sample in &[
            b"".as_slice(),
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"hello world 123!",
        ] {
            let encoded = base64_encode(sample);
            let decoded = base64_decode(&encoded).expect("Decode should succeed");
            assert_eq!(&decoded[..], *sample);
        }
    }

    #[test]
    fn test_encode_png() {
        // 2x2 RGB image
        let rgb_pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let png = encode_png_rgb(2, 2, &rgb_pixels).expect("Encode RGB PNG");
        assert_eq!(
            &png[0..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );

        // 2x2 RGBA image
        let rgba_pixels = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128,
        ];
        let png_rgba = encode_png_rgba(2, 2, &rgba_pixels).expect("Encode RGBA PNG");
        assert_eq!(
            &png_rgba[0..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }
}

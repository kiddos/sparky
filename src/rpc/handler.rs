use crate::animation::FramePacer;
use crate::brain::BrainEngine;
use crate::render::PetWindow;
use crate::rpc::protocol::RpcError;
use crate::state::{EmotionRegistry, StateMachine};
use crate::tray::TrayService;
use crate::ui::ChatState;
use serde_json::json;
use std::sync::Arc;

/// Internal command dispatched to the main thread
pub struct RpcCommand {
    pub method: String,
    pub params: serde_json::Value,
    pub responder: Option<crossbeam_channel::Sender<Result<serde_json::Value, RpcError>>>,
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
            | "get_state"
            | "get_status"
            | "sparky.get_status"
            | "sparky_getStatus"
            | "sparky_getState"
            | "sparky.set_emotion"
            | "set_emotion"
            | "sparky_setEmotion"
            | "sparky.pet"
            | "pet"
            | "sparky_pet"
            | "sparky.poke"
            | "poke"
            | "sparky_poke"
            | "sparky.touch"
            | "touch"
            | "sparky_touch"
            | "sparky.send_chat"
            | "send_chat"
            | "chat"
            | "sparky.chat"
            | "sparky_sendChat"
            | "sparky.get_chat_history"
            | "get_chat_history"
            | "get_messages"
            | "sparky_getChatHistory"
            | "sparky.clear_chat"
            | "clear_chat"
            | "sparky_clearChat"
            | "sparky.set_visible"
            | "set_visible"
            | "sparky_setVisible"
            | "sparky.set_click_through"
            | "set_click_through"
            | "sparky_setClickThrough"
            | "sparky.set_chat_visible"
            | "set_chat_visible"
            | "sparky_setChatVisible"
            | "sparky.set_position"
            | "set_position"
            | "sparky_setPosition"
            | "sparky.get_position"
            | "get_position"
            | "sparky_getPosition"
            | "sparky.quit"
            | "quit"
            | "exit"
            | "sparky_quit"
    )
}

/// Checks if an RPC method can be handled directly by the worker thread without main-thread synchronization
pub fn try_handle_direct(
    method: &str,
    registry: &Arc<EmotionRegistry>,
) -> Option<Result<serde_json::Value, RpcError>> {
    match method {
        "sparky.list_emotions"
        | "list_emotions"
        | "get_emotions"
        | "sparky.get_emotions"
        | "sparky_listEmotions" => {
            let emotions: Vec<_> = registry.all_emotions().collect();
            Some(serde_json::to_value(&emotions).map_err(|e| RpcError::internal(e.to_string())))
        }
        "sparky.version" | "version" | "sparky_version" => Some(Ok(json!({
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
        }))),
        "rpc.discover" | "sparky.list_methods" | "list_methods" | "rpc_discover" => {
            Some(Ok(discover_rpc_methods()))
        }
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
                "aliases": ["get_state", "get_status", "sparky.get_status", "sparky_getStatus"],
                "description": "Returns full desktop pet state including current emotion, active clip, window position and visibility, and chat status",
                "params": [],
                "result": { "description": "Current state object" }
            },
            {
                "name": "sparky.set_emotion",
                "aliases": ["set_emotion", "sparky_setEmotion"],
                "description": "Transitions the pet to a specific emotion (e.g. sit, happy, angry, naughty, sad, smirk, proud)",
                "params": [
                    { "name": "emotion", "description": "Target emotion ID", "required": true, "schema": { "type": "string" } }
                ],
                "result": { "description": "Transition result with resolved clip and shader tint" }
            },
            {
                "name": "sparky.list_emotions",
                "aliases": ["list_emotions", "get_emotions", "sparky.get_emotions", "sparky_listEmotions"],
                "description": "Lists all registered emotions and their configurations (loop mode, duration, fallback, tint)",
                "params": [],
                "result": { "description": "Array of emotion definitions" }
            },
            {
                "name": "sparky.pet",
                "aliases": ["pet", "sparky_pet"],
                "description": "Simulates petting Sparky, transitioning him to 'happy'",
                "params": [],
                "result": { "description": "Result indicating transition to happy" }
            },
            {
                "name": "sparky.poke",
                "aliases": ["poke", "sparky_poke"],
                "description": "Simulates poking Sparky, transitioning him to 'angry'",
                "params": [],
                "result": { "description": "Result indicating transition to angry" }
            },
            {
                "name": "sparky.touch",
                "aliases": ["touch", "sparky_touch"],
                "description": "Registers user interaction, resetting the boredom timer",
                "params": [],
                "result": { "description": "Success confirmation" }
            },
            {
                "name": "sparky.send_chat",
                "aliases": ["send_chat", "chat", "sparky.chat", "sparky_sendChat"],
                "description": "Sends a message/prompt to Sparky's cognitive brain engine",
                "params": [
                    { "name": "prompt", "description": "Message text to send to Sparky", "required": true, "schema": { "type": "string" } }
                ],
                "result": { "description": "Confirmation of queued prompt" }
            },
            {
                "name": "sparky.get_chat_history",
                "aliases": ["get_chat_history", "sparky_getChatHistory"],
                "description": "Returns the conversation message history",
                "params": [],
                "result": { "description": "Array of {sender, message} objects" }
            },
            {
                "name": "sparky.clear_chat",
                "aliases": ["clear_chat", "sparky_clearChat"],
                "description": "Clears conversation history and resets to default greeting",
                "params": [],
                "result": { "description": "Success confirmation" }
            },
            {
                "name": "sparky.set_visible",
                "aliases": ["set_visible", "sparky_setVisible"],
                "description": "Show or hide the desktop pet window",
                "params": [
                    { "name": "visible", "description": "Window visibility state", "required": true, "schema": { "type": "boolean" } }
                ],
                "result": { "description": "Updated visibility state" }
            },
            {
                "name": "sparky.set_click_through",
                "aliases": ["set_click_through", "sparky_setClickThrough"],
                "description": "Enable or disable mouse click-through mode",
                "params": [
                    { "name": "click_through", "description": "Click-through state", "required": true, "schema": { "type": "boolean" } }
                ],
                "result": { "description": "Updated click-through state" }
            },
            {
                "name": "sparky.set_chat_visible",
                "aliases": ["set_chat_visible", "sparky_setChatVisible"],
                "description": "Toggle egui chat overlay visibility",
                "params": [
                    { "name": "visible", "description": "Chat visibility state", "required": true, "schema": { "type": "boolean" } }
                ],
                "result": { "description": "Updated chat visibility state" }
            },
            {
                "name": "sparky.set_position",
                "aliases": ["set_position", "sparky_setPosition"],
                "description": "Sets the desktop position (x, y) of the pet window",
                "params": [
                    { "name": "x", "description": "Screen X position", "required": true, "schema": { "type": "integer" } },
                    { "name": "y", "description": "Screen Y position", "required": true, "schema": { "type": "integer" } }
                ],
                "result": { "description": "Updated window coordinates" }
            },
            {
                "name": "sparky.get_position",
                "aliases": ["get_position", "sparky_getPosition"],
                "description": "Gets current screen position (x, y) of the pet window",
                "params": [],
                "result": { "description": "Object with {x, y}" }
            },
            {
                "name": "sparky.quit",
                "aliases": ["quit", "exit", "sparky_quit"],
                "description": "Gracefully terminates Sparky application",
                "params": [],
                "result": { "description": "Shutdown status" }
            },
            {
                "name": "sparky.version",
                "aliases": ["version", "sparky_version"],
                "description": "Returns application version and metadata",
                "params": [],
                "result": { "description": "Version metadata" }
            },
            {
                "name": "rpc.discover",
                "aliases": ["list_methods", "sparky.list_methods", "rpc_discover"],
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
) -> Result<serde_json::Value, RpcError> {
    match method {
        "sparky.get_state" | "get_state" | "get_status" | "sparky.get_status"
        | "sparky_getStatus" | "sparky_getState" => {
            let (win_x, win_y) = pet_window.window.get_pos();
            let (win_w, win_h) = pet_window.window.get_size();
            Ok(json!({
                "emotion": state_machine.current_emotion,
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
        "sparky.set_emotion" | "set_emotion" | "sparky_setEmotion" => {
            let emotion_id =
                extract_string_param(params, &["emotion", "name", "id"]).ok_or_else(|| {
                    RpcError::invalid_params("Missing or invalid 'emotion' parameter")
                })?;

            if emotion_registry.get(&emotion_id).is_some() {
                state_machine.transition_to(&emotion_id, emotion_registry);
                chat_state.current_emotion_id = emotion_id.clone();
                frame_pacer.reset();
                Ok(json!({
                    "success": true,
                    "emotion": emotion_id,
                    "active_clip": state_machine.active_clip_name,
                    "active_tint": state_machine.active_tint,
                }))
            } else {
                Err(RpcError::invalid_params(format!(
                    "Unknown emotion '{}'. Available: {:?}",
                    emotion_id,
                    emotion_registry.available_emotion_ids()
                )))
            }
        }
        "sparky.pet" | "pet" | "sparky_pet" => {
            state_machine.on_pet(emotion_registry);
            chat_state.current_emotion_id = "happy".to_string();
            frame_pacer.reset();
            Ok(json!({
                "success": true,
                "emotion": state_machine.current_emotion,
            }))
        }
        "sparky.poke" | "poke" | "sparky_poke" => {
            state_machine.on_poke(emotion_registry);
            chat_state.current_emotion_id = "angry".to_string();
            frame_pacer.reset();
            Ok(json!({
                "success": true,
                "emotion": state_machine.current_emotion,
            }))
        }
        "sparky.touch" | "touch" | "sparky_touch" => {
            state_machine.touch_interaction();
            Ok(json!({ "success": true }))
        }
        "sparky.send_chat" | "send_chat" | "chat" | "sparky.chat" | "sparky_sendChat" => {
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
        "sparky.get_chat_history"
        | "get_chat_history"
        | "get_messages"
        | "sparky_getChatHistory" => {
            let history: Vec<_> = chat_state
                .messages
                .iter()
                .map(|(sender, msg)| json!({ "sender": sender, "message": msg }))
                .collect();
            Ok(json!(history))
        }
        "sparky.clear_chat" | "clear_chat" | "sparky_clearChat" => {
            chat_state.messages.clear();
            chat_state.messages.push((
                "Sparky".to_string(),
                "Meow! I'm Sparky. How are you today?".to_string(),
            ));
            Ok(json!({ "success": true }))
        }
        "sparky.set_visible" | "set_visible" | "sparky_setVisible" => {
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
        "sparky.set_click_through" | "set_click_through" | "sparky_setClickThrough" => {
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
        "sparky.set_chat_visible" | "set_chat_visible" | "sparky_setChatVisible" => {
            let visible =
                extract_bool_param(params, &["visible", "value", "enabled"]).ok_or_else(|| {
                    RpcError::invalid_params("Missing or invalid 'visible' boolean parameter")
                })?;

            chat_state.is_visible = visible;
            Ok(json!({ "success": true, "chat_visible": visible }))
        }
        "sparky.set_position" | "set_position" | "sparky_setPosition" => {
            let (x, y) = extract_pos_params(params).ok_or_else(|| {
                RpcError::invalid_params("Missing or invalid coordinates. Provide {x, y} or [x, y]")
            })?;

            pet_window.window.set_pos(x, y);
            Ok(json!({ "success": true, "x": x, "y": y }))
        }
        "sparky.get_position" | "get_position" | "sparky_getPosition" => {
            let (x, y) = pet_window.window.get_pos();
            Ok(json!({ "x": x, "y": y }))
        }
        "sparky.quit" | "quit" | "exit" | "sparky_quit" => {
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
        assert!(methods.iter().any(|m| m["name"] == "sparky.set_emotion"));
        assert!(methods.iter().any(|m| m["name"] == "sparky.send_chat"));
        assert!(methods.iter().any(|m| m["name"] == "sparky.pet"));
    }
}

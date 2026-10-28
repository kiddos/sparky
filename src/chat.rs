use slint::Model;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, Once};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MessageSender {
    User,
    Sparky,
    System,
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub sender: MessageSender,
    pub text: String,
    pub emotion: Option<String>,
}

slint::slint! {
    import { ScrollView, LineEdit } from "std-widgets.slint";

    export struct ChatItem {
        sender_type: int,
        header: string,
        text: string,
    }

    export component ChatWindowView inherits Window {
        width: 380px;
        height: 360px;
        no-frame: true;
        always-on-top: true;
        background: transparent;
        title: "Sparky Chat";

        in property <string> personality_name: "chaos_gremlin";
        in property <string> status_text: "";
        in property <[ChatItem]> messages: [];

        callback send_message(string);
        callback close_clicked();

        public function scroll_to_bottom() {
            scroll.content-y = min(0px, scroll.height - scroll.content-height);
        }

        Rectangle {
            width: 100%;
            height: 100%;
            background: #1e1e2e;
            border-color: #89b4fa;
            border-width: 2px;
            border-radius: 12px;
            clip: true;

            VerticalLayout {
                padding: 0px;
                spacing: 0px;

                // Header Bar (Draggable via WindowMoveArea)
                Rectangle {
                    height: 38px;
                    background: #181825;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;

                    WindowMoveArea {
                        HorizontalLayout {
                            padding-left: 12px;
                            padding-right: 8px;
                            alignment: space-between;

                            HorizontalLayout {
                                spacing: 8px;
                                alignment: start;

                                Text {
                                    text: "✨ Sparky Chat";
                                    color: #cdd6f4;
                                    font-size: 13px;
                                    font-weight: 700;
                                    vertical-alignment: center;
                                }

                                Text {
                                    text: root.status_text != "" ? root.status_text : "(" + root.personality_name + ")";
                                    color: #f9e2af;
                                    font-size: 11px;
                                    font-italic: true;
                                    vertical-alignment: center;
                                }
                            }

                            // Close button "✕"
                            close_btn := Rectangle {
                                width: 24px;
                                height: 24px;
                                border-radius: 4px;
                                background: close_touch.pressed ? #f38ba8.transparentize(80%) : close_touch.has-hover ? #f38ba8.transparentize(90%) : transparent;

                                Text {
                                    text: "✕";
                                    color: close_touch.has-hover ? #f38ba8 : #a6adc8;
                                    font-size: 13px;
                                    font-weight: 700;
                                    horizontal-alignment: center;
                                    vertical-alignment: center;
                                }

                                close_touch := TouchArea {
                                    mouse-cursor: pointer;
                                    clicked => {
                                        root.close_clicked();
                                    }
                                }
                            }
                        }
                    }

                    Rectangle {
                        y: parent.height - 1px;
                        height: 1px;
                        background: #313244;
                    }
                }

                // Messages Scroll Area
                scroll := ScrollView {
                    vertical-stretch: 1;

                    VerticalLayout {
                        padding-left: 10px;
                        padding-right: 10px;
                        padding-top: 10px;
                        padding-bottom: 24px;
                        spacing: 10px;
                        alignment: start;

                        for item in root.messages: VerticalLayout {
                            padding: 0px;
                            spacing: 0px;

                            // User Message (Right-aligned bubble)
                            if item.sender_type == 0: HorizontalLayout {
                                alignment: end;
                                padding-left: 36px;

                                Rectangle {
                                    max-width: root.width - 64px;
                                    background: #3b4261;
                                    border-radius: 8px;

                                    VerticalLayout {
                                        padding-left: 12px;
                                        padding-right: 12px;
                                        padding-top: 7px;
                                        padding-bottom: 7px;
                                        spacing: 3px;

                                        Text {
                                            text: item.header;
                                            color: #89b4fa;
                                            font-size: 11px;
                                            font-weight: 700;
                                        }

                                        Text {
                                            text: item.text;
                                            color: #cdd6f4;
                                            font-size: 13px;
                                            wrap: word-wrap;
                                        }
                                    }
                                }
                            }

                            // Sparky Message (Left-aligned bubble with accent stripe)
                            if item.sender_type == 1: HorizontalLayout {
                                alignment: start;
                                padding-right: 36px;

                                Rectangle {
                                    max-width: root.width - 64px;
                                    background: #2a2b3d;
                                    border-radius: 8px;

                                    HorizontalLayout {
                                        padding: 0px;
                                        spacing: 0px;

                                        // Left accent bar
                                        Rectangle {
                                            width: 4px;
                                            background: #f9e2af;
                                            border-top-left-radius: 8px;
                                            border-bottom-left-radius: 8px;
                                        }

                                        VerticalLayout {
                                            padding-left: 10px;
                                            padding-right: 12px;
                                            padding-top: 7px;
                                            padding-bottom: 7px;
                                            spacing: 3px;

                                            Text {
                                                text: item.header;
                                                color: #f9e2af;
                                                font-size: 11px;
                                                font-weight: 700;
                                            }

                                            Text {
                                                text: item.text;
                                                color: #cdd6f4;
                                                font-size: 13px;
                                                wrap: word-wrap;
                                            }
                                        }
                                    }
                                }
                            }

                            // System Message (Centered italic notification)
                            if item.sender_type == 2: HorizontalLayout {
                                alignment: center;
                                padding-top: 4px;
                                padding-bottom: 4px;
                                padding-left: 16px;
                                padding-right: 16px;

                                Text {
                                    text: item.text;
                                    color: #a6adc8;
                                    font-size: 11px;
                                    font-italic: true;
                                    wrap: word-wrap;
                                    horizontal-alignment: center;
                                }
                            }
                        }
                    }
                }

                // Input Bar
                Rectangle {
                    height: 44px;
                    background: #181825;
                    border-bottom-left-radius: 10px;
                    border-bottom-right-radius: 10px;

                    Rectangle {
                        y: 0px;
                        height: 1px;
                        background: #313244;
                    }

                    HorizontalLayout {
                        padding-left: 10px;
                        padding-right: 10px;
                        padding-top: 7px;
                        padding-bottom: 7px;
                        spacing: 8px;

                        input := LineEdit {
                            horizontal-stretch: 1;
                            placeholder-text: "Say something to Sparky...";
                            font-size: 13px;
                            accepted(text) => {
                                if (text != "") {
                                    root.send_message(text);
                                    self.text = "";
                                }
                            }
                        }

                        send_btn := Rectangle {
                            width: 58px;
                            height: 30px;
                            background: send_touch.pressed ? #74c7ec : send_touch.has-hover ? #b4befe : #89b4fa;
                            border-radius: 6px;

                            Text {
                                text: "Send";
                                color: #11111b;
                                font-size: 12px;
                                font-weight: 700;
                                horizontal-alignment: center;
                                vertical-alignment: center;
                            }

                            send_touch := TouchArea {
                                mouse-cursor: pointer;
                                clicked => {
                                    if (input.text != "") {
                                        root.send_message(input.text);
                                        input.text = "";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

thread_local! {
    static WINDOWS: RefCell<HashMap<u64, ChatWindowView>> = RefCell::new(HashMap::new());
}

static NEXT_WINDOW_ID: AtomicU64 = AtomicU64::new(1);
static START_UI_THREAD: Once = Once::new();
static UI_READY_TX: Mutex<Option<Sender<()>>> = Mutex::new(None);

fn ensure_ui_thread() {
    START_UI_THREAD.call_once(|| {
        let (init_tx, init_rx) = channel::<()>();
        *UI_READY_TX.lock().unwrap() = Some(init_tx);

        std::thread::Builder::new()
            .name("sparky-slint-ui".to_string())
            .spawn(|| {
                let _init = ChatWindowView::new().expect("Failed to initialize Slint");
                if let Some(tx) = UI_READY_TX.lock().unwrap().take() {
                    let _ = tx.send(());
                }
                let _ = slint::run_event_loop_until_quit();
            })
            .expect("Failed to spawn sparky-slint-ui thread");

        let _ = init_rx.recv();
    });
}

pub struct ChatWindow {
    id: u64,
    user_msg_rx: Receiver<String>,
    visible: Arc<AtomicBool>,
    width: i32,
    height: i32,
    initial_docked_pos: Arc<Mutex<Option<(i32, i32)>>>,
}

impl ChatWindow {
    pub fn new(personality_name: &str) -> Self {
        ensure_ui_thread();

        let id = NEXT_WINDOW_ID.fetch_add(1, Ordering::Relaxed);
        let (user_msg_tx, user_msg_rx) = channel::<String>();
        let visible = Arc::new(AtomicBool::new(false));
        let initial_docked_pos = Arc::new(Mutex::new(None));
        let default_w = 380;
        let default_h = 360;

        let personality = personality_name.to_string();
        let vis_clone = visible.clone();
        let (done_tx, done_rx) = channel::<()>();

        slint::invoke_from_event_loop(move || {
            let ui = ChatWindowView::new().expect("Failed to create ChatWindowView");
            ui.set_personality_name(personality.into());

            let tx = user_msg_tx;
            ui.on_send_message(move |text| {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    let _ = tx.send(trimmed.to_string());
                }
            });

            let vis = vis_clone;
            let weak_ui = ui.as_weak();
            ui.on_close_clicked(move || {
                vis.store(false, Ordering::SeqCst);
                if let Some(u) = weak_ui.upgrade() {
                    let _ = u.window().hide();
                }
            });

            WINDOWS.with(|w| {
                w.borrow_mut().insert(id, ui);
            });

            let _ = done_tx.send(());
        })
        .expect("Failed to invoke on Slint event loop");

        let _ = done_rx.recv();

        Self {
            id,
            user_msg_rx,
            visible,
            width: default_w,
            height: default_h,
            initial_docked_pos,
        }
    }

    pub fn set_status(&self, status: &str) {
        let id = self.id;
        let s = status.to_string();
        let _ = slint::invoke_from_event_loop(move || {
            WINDOWS.with(|w| {
                if let Some(ui) = w.borrow().get(&id) {
                    ui.set_status_text(s.into());
                }
            });
        });
    }

    pub fn add_message(&self, message: &ChatMessage) {
        let (sender_type, header) = match message.sender {
            MessageSender::User => (0, "You:".to_string()),
            MessageSender::Sparky => {
                let tag = if let Some(ref em) = message.emotion {
                    format!("Sparky [{}]:", em)
                } else {
                    "Sparky:".to_string()
                };
                (1, tag)
            }
            MessageSender::System => (2, String::new()),
        };
        let text = message.text.clone();
        let id = self.id;

        let _ = slint::invoke_from_event_loop(move || {
            WINDOWS.with(|w| {
                if let Some(ui) = w.borrow().get(&id) {
                    let item = ChatItem {
                        sender_type,
                        header: header.into(),
                        text: text.into(),
                    };
                    let mut items: Vec<ChatItem> = ui.get_messages().iter().collect();
                    items.push(item);
                    ui.set_messages(slint::ModelRc::from(items.as_slice()));
                    ui.invoke_scroll_to_bottom();

                    // Re-invoke on the next event loop tick so newly laid out content height is captured
                    let weak = ui.as_weak();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(u) = weak.upgrade() {
                            u.invoke_scroll_to_bottom();
                        }
                    });
                }
            });
        });
    }

    pub fn add_user_message(&self, text: &str) {
        self.add_message(&ChatMessage {
            sender: MessageSender::User,
            text: text.to_string(),
            emotion: None,
        });
    }

    pub fn add_sparky_message(&self, emotion: &str, text: &str) {
        self.add_message(&ChatMessage {
            sender: MessageSender::Sparky,
            text: text.to_string(),
            emotion: Some(emotion.to_string()),
        });
    }

    pub fn try_recv_user_message(&self) -> Option<String> {
        self.user_msg_rx.try_recv().ok()
    }

    pub fn is_visible(&self) -> bool {
        self.visible.load(Ordering::SeqCst)
    }

    pub fn initial_docked_pos(&self) -> Option<(i32, i32)> {
        *self.initial_docked_pos.lock().unwrap()
    }

    pub fn set_initial_docked_pos(&self, pos: Option<(i32, i32)>) {
        *self.initial_docked_pos.lock().unwrap() = pos;
    }

    pub fn toggle_visibility(
        &mut self,
        sparky_x: i32,
        sparky_y: i32,
        sparky_w: u32,
        sparky_h: u32,
        screen_h: i32,
    ) {
        let target_visible = !self.is_visible();
        self.set_visible(target_visible, sparky_x, sparky_y, sparky_w, sparky_h, screen_h);
    }

    pub fn set_visible(
        &mut self,
        show: bool,
        sparky_x: i32,
        sparky_y: i32,
        sparky_w: u32,
        sparky_h: u32,
        screen_h: i32,
    ) {
        self.visible.store(show, Ordering::SeqCst);
        let default_pos = calculate_chat_position(
            sparky_x,
            sparky_y,
            sparky_w,
            sparky_h,
            self.width,
            self.height,
            screen_h,
        );
        let id = self.id;
        let init_pos = self.initial_docked_pos.clone();
        let _ = slint::invoke_from_event_loop(move || {
            WINDOWS.with(|w| {
                if let Some(ui) = w.borrow().get(&id) {
                    if show {
                        let current_pos = ui.window().position();
                        let mut guard = init_pos.lock().unwrap();
                        let target_pos = if let Some(last_docked) = *guard {
                            // If user dragged the window to a new position, preserve it
                            if (current_pos.x, current_pos.y) != last_docked && (current_pos.x != 0 || current_pos.y != 0) {
                                (current_pos.x, current_pos.y)
                            } else {
                                *guard = Some(default_pos);
                                default_pos
                            }
                        } else {
                            *guard = Some(default_pos);
                            default_pos
                        };

                        ui.window().set_position(slint::PhysicalPosition::new(
                            target_pos.0,
                            target_pos.1,
                        ));
                        let _ = ui.window().show();
                    } else {
                        let _ = ui.window().hide();
                    }
                }
            });
        });
    }

    pub fn update_position(
        &self,
        sparky_x: i32,
        sparky_y: i32,
        sparky_w: u32,
        sparky_h: u32,
        screen_h: i32,
    ) {
        let new_docked_pos = calculate_chat_position(
            sparky_x,
            sparky_y,
            sparky_w,
            sparky_h,
            self.width,
            self.height,
            screen_h,
        );
        let id = self.id;
        let init_pos = self.initial_docked_pos.clone();
        let _ = slint::invoke_from_event_loop(move || {
            WINDOWS.with(|w| {
                if let Some(ui) = w.borrow().get(&id) {
                    let current_pos = ui.window().position();
                    let mut guard = init_pos.lock().unwrap();
                    if let Some(last_docked) = *guard {
                        // If user dragged the window away, preserve the user position
                        if (current_pos.x, current_pos.y) != last_docked && (current_pos.x != 0 || current_pos.y != 0) {
                            return;
                        }
                    }
                    *guard = Some(new_docked_pos);
                    ui.window().set_position(slint::PhysicalPosition::new(
                        new_docked_pos.0,
                        new_docked_pos.1,
                    ));
                }
            });
        });
    }
}

impl Drop for ChatWindow {
    fn drop(&mut self) {
        let id = self.id;
        let _ = slint::invoke_from_event_loop(move || {
            WINDOWS.with(|w| {
                if let Some(ui) = w.borrow_mut().remove(&id) {
                    let _ = ui.window().hide();
                }
            });
        });
    }
}

pub fn calculate_chat_position(
    sparky_x: i32,
    sparky_y: i32,
    sparky_w: u32,
    sparky_h: u32,
    chat_w: i32,
    chat_h: i32,
    screen_h: i32,
) -> (i32, i32) {
    let chat_x = (sparky_x + (sparky_w as i32 - chat_w) / 2).max(10);
    let ideal_y = sparky_y + sparky_h as i32 + 6;
    let chat_y = if screen_h > 0 && ideal_y + chat_h > screen_h {
        (sparky_y - chat_h - 6).max(10)
    } else {
        ideal_y
    };
    (chat_x, chat_y)
}

#[cfg(test)]
fn glib_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_chat_position_below() {
        let (x, y) = calculate_chat_position(500, 200, 200, 200, 360, 260, 1080);
        assert_eq!(x, 420);
        assert_eq!(y, 406);
    }

    #[test]
    fn test_calculate_chat_position_overflow_flips_above() {
        let (x, y) = calculate_chat_position(500, 900, 200, 200, 360, 260, 1080);
        assert_eq!(x, 420);
        assert_eq!(y, 634);
    }

    #[test]
    fn test_calculate_chat_position_clamped_left() {
        let (x, _y) = calculate_chat_position(-50, 200, 200, 200, 360, 260, 1080);
        assert_eq!(x, 10);
    }

    #[test]
    fn test_glib_escape() {
        assert_eq!(glib_escape("hello & <world>"), "hello &amp; &lt;world&gt;");
        assert_eq!(glib_escape("\"quoted\" 'single'"), "&quot;quoted&quot; &apos;single&apos;");
    }

    #[test]
    fn test_chat_window_creation_and_messaging() {
        let mut chat = ChatWindow::new("chaos_gremlin");
        assert!(!chat.is_visible());
        assert_eq!(chat.initial_docked_pos(), None);

        chat.set_status("[mischievous]");
        chat.add_message(&ChatMessage {
            sender: MessageSender::System,
            text: "System initialized".to_string(),
            emotion: None,
        });
        chat.add_user_message("Hello Sparky!");
        chat.add_sparky_message("mischievous", "Mwahaha!");

        chat.toggle_visibility(100, 100, 128, 128, 1080);
        assert!(chat.is_visible());

        chat.update_position(120, 150, 128, 128, 1080);

        chat.set_initial_docked_pos(Some((600, 400)));
        assert_eq!(chat.initial_docked_pos(), Some((600, 400)));

        chat.toggle_visibility(100, 100, 128, 128, 1080);
        assert!(!chat.is_visible());
    }

    #[test]
    fn test_chat_message_sender_formatting() {
        let user_msg = ChatMessage {
            sender: MessageSender::User,
            text: "Hello".to_string(),
            emotion: None,
        };
        assert_eq!(user_msg.sender, MessageSender::User);

        let sparky_msg = ChatMessage {
            sender: MessageSender::Sparky,
            text: "Hi there!".to_string(),
            emotion: Some("happy1".to_string()),
        };
        assert_eq!(sparky_msg.sender, MessageSender::Sparky);
        assert_eq!(sparky_msg.emotion.as_deref(), Some("happy1"));

        let sys_msg = ChatMessage {
            sender: MessageSender::System,
            text: "Welcome".to_string(),
            emotion: None,
        };
        assert_eq!(sys_msg.sender, MessageSender::System);
    }

    #[test]
    fn test_chat_with_glfw_integration() {
        #[cfg(target_os = "linux")]
        unsafe {
            unsafe extern "C" {
                fn XInitThreads() -> i32;
            }
            XInitThreads();
        }

        use glfw::Context;
        let mut glfw = match glfw::init(glfw::fail_on_errors) {
            Ok(g) => g,
            Err(_) => return,
        };
        glfw.window_hint(glfw::WindowHint::Resizable(false));
        glfw.window_hint(glfw::WindowHint::Decorated(false));
        glfw.window_hint(glfw::WindowHint::Visible(false));
        let (mut window, _) = match glfw.create_window(128, 128, "Test", glfw::WindowMode::Windowed) {
            Some(w) => w,
            None => return,
        };
        window.make_current();

        let mut chat = ChatWindow::new("chaos_gremlin");
        chat.toggle_visibility(100, 100, 128, 128, 1080);
        assert!(chat.is_visible());

        for _ in 0..10 {
            glfw.poll_events();
        }

        chat.toggle_visibility(100, 100, 128, 128, 1080);
        assert!(!chat.is_visible());

        for _ in 0..10 {
            glfw.poll_events();
        }

        glfw::make_context_current(None);
    }
}

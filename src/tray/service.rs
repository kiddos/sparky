use std::collections::HashMap;
use std::path::Path;
use muda::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
use crate::state::emotion::EmotionRegistry;

pub struct TrayService {
    _tray_icon: TrayIcon,
    pub show_hide_item: CheckMenuItem,
    pub chat_item: MenuItem,
    pub click_through_item: CheckMenuItem,
    pub quit_item: MenuItem,
    pub emotion_menu_items: HashMap<muda::MenuId, String>, // MenuId -> emotion_id
}

impl TrayService {
    pub fn new(registry: &EmotionRegistry) -> anyhow::Result<Self> {
        // Initialize GTK backend required by tray-icon on Linux
        #[cfg(target_os = "linux")]
        if let Err(e) = gtk::init() {
            log::warn!(
                "GTK initialization warning (tray icon may not function if required by platform): {e}"
            );
        }

        let menu = Menu::new();

        let show_hide_item = CheckMenuItem::new("Show Sparky", true, true, None);
        let chat_item = MenuItem::new("Chat with Sparky", true, None);
        let click_through_item = CheckMenuItem::new("Click-Through Mode", true, false, None);
        let quit_item = MenuItem::new("Exit Sparky", true, None);

        // Dynamically build "Pet Mood" submenu from registered emotions
        let mood_submenu = Submenu::new("🐾 Pet Mood", true);
        let mut emotion_menu_items = HashMap::new();

        for emotion in registry.all_emotions() {
            let label = format!("{} {}", emotion.emoji, emotion.name);
            let item = MenuItem::new(&label, true, None);
            emotion_menu_items.insert(item.id().clone(), emotion.id.clone());
            mood_submenu.append(&item)?;
        }

        menu.append(&show_hide_item)?;
        menu.append(&chat_item)?;
        menu.append(&mood_submenu)?;
        menu.append(&click_through_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit_item)?;

        // Attempt to load custom icon if exists, else construct a 32x32 RGBA orange kitten icon
        let icon = Self::load_or_create_icon("assets/icons/sparky_tray.png")?;

        let tray_icon = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("Sparky the Kitten")
            .with_icon(icon)
            .build()?;

        Ok(Self {
            _tray_icon: tray_icon,
            show_hide_item,
            chat_item,
            click_through_item,
            quit_item,
            emotion_menu_items,
        })
    }

    /// Pump the system tray event loop to handle user menu interactions and avoid freezing.
    pub fn pump_events(&self) {
        Self::pump();
    }

    /// Static helper to pump the underlying system tray / GTK event loop.
    pub fn pump() {
        #[cfg(target_os = "linux")]
        while gtk::events_pending() {
            gtk::main_iteration();
        }
    }

    fn load_or_create_icon<P: AsRef<Path>>(icon_path: P) -> anyhow::Result<Icon> {
        if let Ok(img) = image::open(icon_path) {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            return Ok(Icon::from_rgba(rgba.into_raw(), w, h)?);
        }

        // Generate a 32x32 cute orange kitten icon
        let w = 32u32;
        let h = 32u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                let dx = (x as f32) - 16.0;
                let dy = (y as f32) - 16.0;
                let dist = (dx * dx + dy * dy).sqrt();

                // Cat face circular body
                if dist <= 12.0 {
                    rgba[idx] = 255;     // R
                    rgba[idx + 1] = 140; // G
                    rgba[idx + 2] = 0;   // B
                    rgba[idx + 3] = 255; // A
                }
                // Cat ears
                else if (y < 12) && ((x >= 6 && x <= 12) || (x >= 20 && x <= 26)) {
                    rgba[idx] = 240;
                    rgba[idx + 1] = 110;
                    rgba[idx + 2] = 0;
                    rgba[idx + 3] = 255;
                }
                // Cat eyes
                if (y >= 13 && y <= 15) && ((x >= 10 && x <= 12) || (x >= 20 && x <= 22)) {
                    rgba[idx] = 30;
                    rgba[idx + 1] = 30;
                    rgba[idx + 2] = 30;
                    rgba[idx + 3] = 255;
                }
            }
        }

        Ok(Icon::from_rgba(rgba, w, h)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tray_pump_does_not_panic() {
        #[cfg(target_os = "linux")]
        let _ = gtk::init();
        TrayService::pump();
    }
}

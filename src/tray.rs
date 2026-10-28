use log::{info, warn};
use tray_icon::menu::{Menu, MenuId, MenuItem};
use tray_icon::{TrayIcon, TrayIconBuilder};

pub const SHOW_MENU_ID: &str = "SHOW_MENU";
pub const TOGGLE_CHAT_MENU_ID: &str = "TOGGLE_CHAT_MENU";
pub const QUIT_MENU_ID: &str = "QUIT_MENU";

const ICON: &[u8] = include_bytes!("../sparky.png");

pub fn create_tray_icon() -> Option<TrayIcon> {
    info!("create tray icon...");
    let show_menu_item = MenuItem::with_id(MenuId::new(SHOW_MENU_ID), "Show", true, None);
    let chat_menu_item = MenuItem::with_id(MenuId::new(TOGGLE_CHAT_MENU_ID), "Toggle Chat", true, None);
    let quit_menu_item = MenuItem::with_id(MenuId::new(QUIT_MENU_ID), "Quit", true, None);
    let menu = Menu::new();
    let _ = menu.append(&show_menu_item);
    let _ = menu.append(&chat_menu_item);
    let _ = menu.append(&quit_menu_item);

    match TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Sparky")
        .with_icon(load_icon())
        .build()
    {
        Ok(tray_icon) => {
            let _ = tray_icon.set_visible(true);
            Some(tray_icon)
        }
        Err(e) => {
            warn!("Failed to create tray icon: {}", e);
            None
        }
    }
}

fn load_icon() -> tray_icon::Icon {
    info!("load icon...");
    let image = image::load_from_memory(ICON)
        .expect("Failed to load icon image")
        .to_rgba8();
    let image = image::imageops::resize(&image, 256, 256, image::imageops::FilterType::Gaussian);
    let (img_width, img_height) = image.dimensions();
    let image_data = image.into_raw();
    tray_icon::Icon::from_rgba(image_data, img_width, img_height).expect("Failed to create icon")
}

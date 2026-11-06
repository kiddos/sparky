//! Platform-specific integrations and subsystem helpers.

#[cfg(target_os = "windows")]
unsafe extern "system" {
    fn AttachConsole(dwProcessId: u32) -> i32;
}

/// Initializes platform-specific environment settings.
///
/// On Windows, attempts to attach to the parent terminal console if invoked from CLI
/// (e.g. PowerShell or cmd.exe), ensuring `--help`, `--profile-run`, and logs are visible
/// while allowing GUI double-click launches to stay clean without an unwanted black console window.
pub fn init_platform() {
    #[cfg(target_os = "windows")]
    unsafe {
        const ATTACH_PARENT_PROCESS: u32 = 0xFFFFFFFF;
        let _ = AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

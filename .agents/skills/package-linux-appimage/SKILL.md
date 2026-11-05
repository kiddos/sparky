---
name: package-linux-appimage
description: >-
  Use this skill when packaging Linux executables (Rust, C/C++, Go, or generic binaries)
  into standalone AppImages, creating AppDir hierarchies, writing AppRun entrypoints,
  generating FreeDesktop icons and desktop files, integrating appimagetool, or verifying
  and testing AppImage bundles.
---

# Linux Standalone Executable AppImage Packaging Runbook

This guide details the end-to-end workflow for packaging standalone Linux executables into universal, self-contained AppImage bundles.

---

## 1. Architectural Invariants & Graphics Boundaries

> [!IMPORTANT]
> **Never bundle host display and graphics driver libraries.**
> The following shared libraries must resolve dynamically to the host system:
> - Hardware OpenGL / Vulkan / DRI drivers: `libGL.so*`, `libGLX*`, `libEGL.so*`, `libdrm.so*`, `libvulkan.so*`
> - Display server primitives: `libX11.so*`, `libxcb.so*`, `libwayland-client.so*`
> - Core C runtime / threading: `libc.so*`, `libm.so*`, `libpthread.so*`, `libdl.so*`, `ld-linux*`
> Bundling these creates ABI mismatch panics with host GPU drivers and disables hardware acceleration.

---

## 2. Standard AppDir Directory Hierarchy

Before packing with `appimagetool`, assemble the following directory structure:

```text
AppDir/
├── AppRun                                     # Executable POSIX shell entrypoint (chmod +x)
├── <app_id>.desktop                           # FreeDesktop application shortcut
├── <app_id>.png                               # Root application icon (256x256 PNG)
├── assets/                                    # (Optional) Bundled application assets
├── usr/
│   ├── bin/
│   │   └── <binary_name>                      # Target executable binary (chmod +x)
│   ├── lib/                                   # (Optional) Bundled third-party shared libraries
│   └── share/
│       ├── applications/<app_id>.desktop
│       ├── icons/hicolor/256x256/apps/<app_id>.png
│       └── metainfo/<app_id>.appdata.xml      # AppStream metadata
```

---

## 3. Production `AppRun` Entrypoint Template

Save as `AppDir/AppRun` and mark executable (`chmod +x AppDir/AppRun`):

```bash
#!/usr/bin/env bash
set -e

# Resolve the absolute mount directory of the AppImage
HERE="$(dirname "$(readlink -f "${0}")")"
CALLER_DIR="${OWD:-$PWD}"

export APPDIR="${HERE}"
export PATH="${HERE}/usr/bin:${PATH}"
if [ -d "${HERE}/usr/lib" ]; then
    export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH:-}"
fi
export XDG_DATA_DIRS="${HERE}/usr/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"

# If bundled assets exist in AppDir, but not in caller's directory,
# switch working directory to AppDir so relative asset paths resolve seamlessly.
if [ -d "${HERE}/shaders" ] && [ ! -d "${CALLER_DIR}/shaders" ]; then
    cd "${HERE}"
elif [ -d "${HERE}/assets" ] && [ ! -d "${CALLER_DIR}/assets" ]; then
    cd "${HERE}"
fi

exec "${HERE}/usr/bin/@BIN_NAME@" "$@"
```

---

## 4. AppStream Metadata & Desktop Specification

### Desktop File (`AppDir/<app_id>.desktop`)
```ini
[Desktop Entry]
Type=Application
Name=@APP_NAME@
Exec=@BIN_NAME@ %U
Icon=@APP_ID@
Categories=Utility;
Terminal=false
StartupNotify=true
```

### AppStream Metainfo (`AppDir/usr/share/metainfo/<app_id>.appdata.xml`)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
  <id>org.desktop.@APP_ID@</id>
  <metadata_license>CC0-1.0</metadata_license>
  <project_license>MIT</project_license>
  <name>@APP_NAME@</name>
  <summary>@APP_NAME@ packaged as a standalone AppImage</summary>
  <description>
    <p>@APP_NAME@ application packaged as an AppImage bundle.</p>
  </description>
  <provides>
    <binary>@BIN_NAME@</binary>
  </provides>
</component>
```

---

## 5. Zero-Dependency 256x256 PNG Icon Generator

If no application icon is available, generate a valid FreeDesktop PNG icon using Python standard library:

```bash
python3 -c "
import struct, zlib, math, os, sys
target_file = sys.argv[1]
size = 256
rows = []
center = size / 2.0
radius = size * 0.44
for y in range(size):
    row = bytearray()
    for x in range(size):
        dx, dy = x - center, y - center
        d = math.sqrt(dx*dx + dy*dy)
        if d <= radius:
            factor = (y / size)
            r, g, b, a = int(40 + 20 * factor), int(120 + 40 * factor), int(220 + 35 * factor), 255
            if radius - d < 3.0:
                a = int(255 * (radius - d) / 3.0)
            if d > radius - 8.0:
                r, g, b = min(255, r + 40), min(255, g + 40), min(255, b + 40)
        else:
            r, g, b, a = 0, 0, 0, 0
        row.extend([r, g, b, a])
    rows.append(bytes(row))
raw_data = b''.join(b'\x00' + r for r in rows)
def chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))
png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(raw_data, 9))
png += chunk(b'IEND', b'')
with open(target_file, 'wb') as f:
    f.write(png)
" "AppDir/<app_id>.png"
```

---

## 6. Downloading & Invoking `appimagetool`

```bash
ARCH="$(uname -m)"
export ARCH

# 1. Download official appimagetool if not on PATH
TOOL_PATH="${HOME}/.cache/appimage/appimagetool-${ARCH}.AppImage"
if [ ! -x "$TOOL_PATH" ] && ! command -v appimagetool &>/dev/null; then
    mkdir -p "$(dirname "$TOOL_PATH")"
    curl -sSL -f -o "$TOOL_PATH" \
      "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
    chmod +x "$TOOL_PATH"
fi
TOOL="${TOOL_PATH:-appimagetool}"

# 2. Check FUSE execution capability; fallback to --appimage-extract-and-run
TOOL_CMD=("$TOOL")
if ! "$TOOL" --version &>/dev/null; then
    TOOL_CMD=("$TOOL" "--appimage-extract-and-run")
fi

# 3. Build AppImage (pass -n / --no-appstream to avoid strict metadata failures)
"${TOOL_CMD[@]}" -n "AppDir" "Output-${ARCH}.AppImage"
chmod +x "Output-${ARCH}.AppImage"
```

---

## 7. Automated Verification Protocol

Always verify the output artifact before completing a packaging task:

1. **Header & Magic Bytes Check:**
   ```bash
   file "Output-x86_64.AppImage" | grep -q "ELF"
   dd if="Output-x86_64.AppImage" bs=1 skip=8 count=3 2>/dev/null | grep -q "AI"
   ```
2. **Help & Version Invocation:**
   ```bash
   ./Output-x86_64.AppImage --version || ./Output-x86_64.AppImage --appimage-extract-and-run --version
   ```
3. **Out-of-Tree Asset Isolation Test:**
   ```bash
   (cd /tmp && /path/to/Output-x86_64.AppImage --help)
   ```
4. **Reference Implementation:**
   See [`scripts/package_appimage.sh`](file:///home/kiddos/projects/tools/sparky/scripts/package_appimage.sh) for an automated, production-tested implementation.

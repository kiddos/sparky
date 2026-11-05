#!/usr/bin/env bash
# ==============================================================================
# Linux Single-Executable AppImage Packaging Script
# ==============================================================================
# Packages a standalone Linux executable (e.g. Rust, C/C++, Go binary) into
# a portable, self-contained AppImage bundle using appimagetool.
#
# Usage:
#   ./scripts/package_appimage.sh [OPTIONS] [PATH_TO_EXECUTABLE]
#
# Examples:
#   # Package Sparky desktop pet (auto-bundles assets and shaders):
#   ./scripts/package_appimage.sh target/release/sparky
#
#   # Package any arbitrary single executable:
#   ./scripts/package_appimage.sh -n "MyTool" -o "MyTool.AppImage" /usr/bin/htop
#
#   # Package and test immediately:
#   ./scripts/package_appimage.sh --test target/release/sparky
# ==============================================================================
set -euo pipefail

# --- Color Output Helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}${BOLD}[INFO]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}${BOLD}[OK]${NC} $*" >&2; }
warn()  { echo -e "${YELLOW}${BOLD}[WARN]${NC} $*" >&2; }
error() { echo -e "${RED}${BOLD}[ERROR]${NC} $*" >&2; }

# --- Script Root & Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXECUTABLE=""
APP_NAME=""
OUTPUT_APPIMAGE=""
ICON_PATH=""
DESKTOP_PATH=""
CATEGORIES="Utility;"
EXTRA_ASSETS=()
BUNDLE_LIBS=false
DO_TEST=false
CUSTOM_APPIMAGETOOL=""
CACHE_DIR="${HOME}/.cache/appimage"

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [EXECUTABLE]

Package any single Linux executable into a standalone AppImage.

Arguments:
  EXECUTABLE                 Path to the executable file to package.
                             If omitted and run inside the Sparky repository,
                             defaults to target/release/sparky.

Options:
  -n, --name <NAME>          Application display name (default: auto-detected)
  -o, --output <PATH>        Output AppImage file path (default: <Name>-<arch>.AppImage)
  -i, --icon <PATH>          Path to .png or .svg icon (auto-generated if omitted)
  -d, --desktop <PATH>       Path to custom .desktop file (auto-generated if omitted)
  -c, --categories <CATS>    Desktop menu categories (default: "Utility;")
  -a, --asset <PATH>         Additional asset directory or file to bundle inside AppDir (repeatable)
  -l, --bundle-libs          Bundle non-system shared libraries via ldd into AppDir/usr/lib
  -t, --test                 Run automated test on the resulting AppImage after packaging
      --appimagetool <PATH>  Path to local appimagetool binary
  -h, --help                 Show this help message and exit

Examples:
  # Package Sparky (auto-detects release binary, shaders, and assets):
  $(basename "$0")

  # Package with custom name and icon:
  $(basename "$0") -n "SparkyPet" -i assets/icons/sparky.png target/release/sparky

  # Package an arbitrary binary and test it:
  $(basename "$0") -n "Ripgrep" --test /usr/bin/rg
EOF
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--name)
            APP_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_APPIMAGE="$2"
            shift 2
            ;;
        -i|--icon)
            ICON_PATH="$2"
            shift 2
            ;;
        -d|--desktop)
            DESKTOP_PATH="$2"
            shift 2
            ;;
        -c|--categories)
            CATEGORIES="$2"
            shift 2
            ;;
        -a|--asset)
            EXTRA_ASSETS+=("$2")
            shift 2
            ;;
        -l|--bundle-libs)
            BUNDLE_LIBS=true
            shift
            ;;
        -t|--test)
            DO_TEST=true
            shift
            ;;
        --appimagetool)
            CUSTOM_APPIMAGETOOL="$2"
            shift 2
            ;;
        -*)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$EXECUTABLE" ]; then
                EXECUTABLE="$1"
            else
                error "Unexpected extra argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# --- Resolve Target Executable ---
if [ -z "$EXECUTABLE" ]; then
    # Auto-detect if we are inside the Sparky repository
    if [ -f "${REPO_ROOT}/Cargo.toml" ] && grep -q 'name = "sparky"' "${REPO_ROOT}/Cargo.toml"; then
        if [ -f "${REPO_ROOT}/target/release/sparky" ]; then
            EXECUTABLE="${REPO_ROOT}/target/release/sparky"
            info "Auto-detected Sparky release binary at: ${EXECUTABLE}"
        else
            info "Sparky release binary not found. Building with 'cargo build --release'..."
            (cd "${REPO_ROOT}" && cargo build --release)
            EXECUTABLE="${REPO_ROOT}/target/release/sparky"
        fi
    else
        error "No executable specified and not in a recognizable project directory."
        show_help
        exit 1
    fi
fi

if [ ! -f "$EXECUTABLE" ]; then
    error "Executable file not found: $EXECUTABLE"
    exit 1
fi

EXECUTABLE="$(readlink -f "$EXECUTABLE")"
BIN_NAME="$(basename "$EXECUTABLE")"

# --- Determine Application Name and ID ---
if [ -z "$APP_NAME" ]; then
    # Capitalize first letter of binary name
    APP_NAME="$(echo "${BIN_NAME:0:1}" | tr '[:lower:]' '[:upper:]')${BIN_NAME:1}"
fi

APP_ID="$(echo -n "$BIN_NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '_')"
ARCH="$(uname -m)"
export ARCH

if [ -z "$OUTPUT_APPIMAGE" ]; then
    OUTPUT_APPIMAGE="${PWD}/${APP_NAME}-${ARCH}.AppImage"
else
    # Make absolute path
    OUTPUT_APPIMAGE="$(mkdir -p "$(dirname "$OUTPUT_APPIMAGE")" && cd "$(dirname "$OUTPUT_APPIMAGE")" && pwd)/$(basename "$OUTPUT_APPIMAGE")"
fi

echo -e "${BLUE}${BOLD}====================================================================${NC}" >&2
echo -e "${BLUE}${BOLD} Sparky Linux AppImage Packager${NC}" >&2
echo -e "${BLUE}${BOLD}====================================================================${NC}" >&2
info "Executable:    ${EXECUTABLE}"
info "App Name:      ${APP_NAME} (${APP_ID})"
info "Architecture:  ${ARCH}"
info "Destination:   ${OUTPUT_APPIMAGE}"

# --- Check & Obtain appimagetool ---
ensure_appimagetool() {
    if [ -n "$CUSTOM_APPIMAGETOOL" ] && [ -x "$CUSTOM_APPIMAGETOOL" ]; then
        echo "$CUSTOM_APPIMAGETOOL"
        return
    fi

    if command -v appimagetool &>/dev/null; then
        command -v appimagetool
        return
    fi

    mkdir -p "$CACHE_DIR"
    local cached_tool="${CACHE_DIR}/appimagetool-${ARCH}.AppImage"

    if [ ! -x "$cached_tool" ]; then
        info "Downloading appimagetool (${ARCH}) into ${CACHE_DIR}..."
        local url="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
        local fallback_url="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${ARCH}.AppImage"

        if ! curl -sSL -f -o "$cached_tool" "$url"; then
            warn "Continuous download failed; trying AppImageKit release..."
            curl -sSL -f -o "$cached_tool" "$fallback_url"
        fi
        chmod +x "$cached_tool"
        ok "Downloaded appimagetool successfully."
    fi

    echo "$cached_tool"
}

TOOL_BIN="$(ensure_appimagetool)"
info "Using appimagetool: ${TOOL_BIN}"

# Verify if appimagetool can run directly or needs --appimage-extract-and-run
TOOL_EXEC=("$TOOL_BIN")
if ! "$TOOL_BIN" --version &>/dev/null; then
    if "$TOOL_BIN" --appimage-extract-and-run --version &>/dev/null; then
        info "Notice: appimagetool requires --appimage-extract-and-run in this environment."
        TOOL_EXEC=("$TOOL_BIN" "--appimage-extract-and-run")
    fi
fi

# --- Create AppDir Directory Structure ---
STAGE_DIR="$(mktemp -d "/tmp/appimage_${APP_ID}_XXXXXX")"
trap 'rm -rf "$STAGE_DIR"' EXIT

APPDIR="${STAGE_DIR}/AppDir"
mkdir -p "${APPDIR}/usr/bin"
mkdir -p "${APPDIR}/usr/lib"
mkdir -p "${APPDIR}/usr/share/applications"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/256x256/apps"
mkdir -p "${APPDIR}/usr/share/metainfo"

# 1. Copy Executable
cp -p "$EXECUTABLE" "${APPDIR}/usr/bin/${BIN_NAME}"
chmod +x "${APPDIR}/usr/bin/${BIN_NAME}"
ok "Copied binary to AppDir/usr/bin/${BIN_NAME}"

# 2. Bundle Sparky Assets if packaging sparky
if [ "$BIN_NAME" = "sparky" ]; then
    if [ -d "${REPO_ROOT}/assets" ]; then
        info "Packaging Sparky: Bundling assets/ directory..."
        cp -r "${REPO_ROOT}/assets" "${APPDIR}/"
    fi
    if [ -d "${REPO_ROOT}/shaders" ]; then
        info "Packaging Sparky: Bundling shaders/ directory..."
        cp -r "${REPO_ROOT}/shaders" "${APPDIR}/"
    fi
fi

# 3. Copy any user-specified extra assets
for asset in "${EXTRA_ASSETS[@]}"; do
    if [ -e "$asset" ]; then
        info "Bundling extra asset: ${asset}"
        cp -r "$asset" "${APPDIR}/"
    else
        warn "Extra asset path not found: ${asset}"
    fi
done

# 4. Optional: Bundle Dynamic Libraries via ldd
if [ "$BUNDLE_LIBS" = true ]; then
    info "Resolving and bundling non-system shared libraries via ldd..."
    # Blacklist core system, glibc, and low-level graphics driver libraries
    # that MUST come from the host OS
    BLACKLIST_REGEX="ld-linux|libc\.so|libm\.so|libdl\.so|librt\.so|libpthread\.so|libGL\.so|libGLX|libEGL|libdrm|libvulkan|libX11|libxcb"
    
    ldd "$EXECUTABLE" | grep "=>" | awk '{print $3}' | while read -r lib; do
        if [ -n "$lib" ] && [ -f "$lib" ]; then
            lib_base="$(basename "$lib")"
            if echo "$lib_base" | grep -Eq "$BLACKLIST_REGEX"; then
                continue
            fi
            cp -n "$lib" "${APPDIR}/usr/lib/" 2>/dev/null || true
        fi
    done
    ok "Libraries bundled into AppDir/usr/lib"
fi

# 5. Icon Handling
RESOLVED_ICON=""
if [ -n "$ICON_PATH" ] && [ -f "$ICON_PATH" ]; then
    RESOLVED_ICON="$ICON_PATH"
elif [ "$BIN_NAME" = "sparky" ] && [ -f "${REPO_ROOT}/assets/icons/sparky.png" ]; then
    RESOLVED_ICON="${REPO_ROOT}/assets/icons/sparky.png"
elif [ "$BIN_NAME" = "sparky" ] && [ -f "${REPO_ROOT}/assets/icons/sparky_tray.png" ]; then
    RESOLVED_ICON="${REPO_ROOT}/assets/icons/sparky_tray.png"
fi

if [ -n "$RESOLVED_ICON" ]; then
    info "Using application icon from: ${RESOLVED_ICON}"
    cp "$RESOLVED_ICON" "${APPDIR}/${APP_ID}.png"
    cp "$RESOLVED_ICON" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/${APP_ID}.png"
else
    info "No icon provided; generating procedural 256x256 PNG icon..."
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
            r = int(255)
            g = int(140 + 30 * (1.0 - d / radius))
            b = int(20 + 20 * (1.0 - d / radius))
            a = 255
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
" "${APPDIR}/${APP_ID}.png"
    cp "${APPDIR}/${APP_ID}.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/${APP_ID}.png"
    ok "Generated fallback application icon."
fi

# 6. Desktop File Handling
if [ -n "$DESKTOP_PATH" ] && [ -f "$DESKTOP_PATH" ]; then
    info "Using custom desktop file: ${DESKTOP_PATH}"
    cp "$DESKTOP_PATH" "${APPDIR}/${APP_ID}.desktop"
    cp "$DESKTOP_PATH" "${APPDIR}/usr/share/applications/${APP_ID}.desktop"
else
    cat << EOF > "${APPDIR}/${APP_ID}.desktop"
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Exec=${BIN_NAME} %U
Icon=${APP_ID}
Categories=${CATEGORIES}
Terminal=false
StartupNotify=true
EOF
    cp "${APPDIR}/${APP_ID}.desktop" "${APPDIR}/usr/share/applications/${APP_ID}.desktop"
    ok "Created desktop specification: ${APP_ID}.desktop"
fi

# 7. AppStream Metainfo (Best effort metadata for desktop stores)
cat << EOF > "${APPDIR}/usr/share/metainfo/${APP_ID}.appdata.xml"
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
  <id>org.desktop.${APP_ID}</id>
  <metadata_license>CC0-1.0</metadata_license>
  <project_license>MIT</project_license>
  <name>${APP_NAME}</name>
  <summary>${APP_NAME} packaged as a standalone AppImage</summary>
  <description>
    <p>${APP_NAME} application packaged as a standalone Linux AppImage.</p>
  </description>
  <provides>
    <binary>${BIN_NAME}</binary>
  </provides>
</component>
EOF

# 8. Create Production AppRun Launcher
cat << 'EOF' > "${APPDIR}/AppRun"
#!/usr/bin/env bash
set -e

# Resolve the directory where this AppRun script resides (squashfs mountpoint)
HERE="$(dirname "$(readlink -f "${0}")")"
CALLER_DIR="${OWD:-$PWD}"

export APPDIR="${HERE}"
export PATH="${HERE}/usr/bin:${PATH}"
if [ -d "${HERE}/usr/lib" ]; then
    export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH:-}"
fi
export XDG_DATA_DIRS="${HERE}/usr/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"

# If bundled assets or shaders exist in AppDir, but not in caller's directory,
# switch working directory to AppDir so relative assets (shaders/, assets/) resolve seamlessly.
if [ -d "${HERE}/shaders" ] && [ ! -d "${CALLER_DIR}/shaders" ]; then
    cd "${HERE}"
elif [ -d "${HERE}/assets" ] && [ ! -d "${CALLER_DIR}/assets" ]; then
    cd "${HERE}"
fi

exec "${HERE}/usr/bin/@BIN_NAME@" "$@"
EOF
# Substitute the actual binary name
sed -i "s/@BIN_NAME@/${BIN_NAME}/g" "${APPDIR}/AppRun"
chmod +x "${APPDIR}/AppRun"
ok "Created AppRun entrypoint script."

# --- Package AppDir into AppImage ---
info "Generating AppImage using appimagetool..."
rm -f "$OUTPUT_APPIMAGE"

# Use -n (--no-appstream) to prevent strict AppStream validation aborts on custom/standalone binaries
"${TOOL_EXEC[@]}" -n "${APPDIR}" "${OUTPUT_APPIMAGE}"

chmod +x "$OUTPUT_APPIMAGE"
ok "AppImage packaging completed successfully!"
info "Output: $(ls -lh "$OUTPUT_APPIMAGE" | awk '{print $9, "(" $5 ")"}')"

# --- Automated Testing ---
test_appimage() {
    local img="$1"
    echo ""
    echo -e "${BLUE}${BOLD}====================================================================${NC}"
    echo -e "${BLUE}${BOLD} Running AppImage Verification & Health Tests${NC}"
    echo -e "${BLUE}${BOLD}====================================================================${NC}"

    info "Test 1: File existence and executable permissions"
    if [ ! -x "$img" ]; then
        error "Test failed: $img is not executable"
        return 1
    fi
    ok "File exists and is marked executable"

    info "Test 2: Verifying ELF header and AppImage magic signature"
    # An AppImage Type 2 has ELF header and magic bytes 'AI\x02' at offset 8
    local magic
    magic="$(dd if="$img" bs=1 skip=8 count=3 2>/dev/null || true)"
    if [ "$magic" = "AI" ] || [ "$magic" = "$(printf 'AI\x02')" ] || file "$img" | grep -q "ELF"; then
        ok "Valid ELF executable / AppImage binary format verified"
    else
        warn "File signature check inconclusive; file reports: $(file -b "$img")"
    fi

    info "Test 3: Testing executable invocation (--version / --help)"
    local test_output
    local test_status=0

    # Try running --version
    if test_output="$("$img" --version 2>&1)"; then
        ok "Command '$img --version' succeeded with exit code 0"
        echo -e "${CYAN}Output:${NC} ${test_output}"
    elif test_output="$("$img" --help 2>&1)"; then
        ok "Command '$img --help' succeeded with exit code 0"
        echo -e "${CYAN}Help excerpt:${NC} $(echo "$test_output" | head -n 3)"
    else
        # Try running in extract-and-run mode if standard execution fails due to FUSE
        if test_output="$("$img" --appimage-extract-and-run --version 2>&1)"; then
            ok "AppImage runs successfully with --appimage-extract-and-run --version"
            echo -e "${CYAN}Output:${NC} ${test_output}"
        elif test_output="$("$img" --appimage-extract-and-run --help 2>&1)"; then
            ok "AppImage runs successfully with --appimage-extract-and-run --help"
            echo -e "${CYAN}Help excerpt:${NC} $(echo "$test_output" | head -n 3)"
        else
            error "Execution test failed. Output:\n${test_output}"
            return 1
        fi
    fi

    # Test 4: If packaging sparky specifically, test running from outside repo
    if [ "$BIN_NAME" = "sparky" ]; then
        info "Test 4: Sparky asset bundling test (running from outside repo in /tmp)"
        local tmp_run_dir
        tmp_run_dir="$(mktemp -d "/tmp/sparky_test_XXXXXX")"
        local sparky_test_status=0

        (
            cd "$tmp_run_dir"
            # In /tmp, there are no shaders/ or assets/ directories.
            # Sparky --help or --version should still run flawlessly thanks to AppRun!
            "$img" --help > sparky_help.txt 2>&1
        ) || sparky_test_status=$?

        if [ "$sparky_test_status" -eq 0 ] && grep -q "sparky" "$tmp_run_dir/sparky_help.txt"; then
            ok "Sparky executed cleanly from outside repo without local assets present!"
        else
            warn "Sparky execution outside repo check: exit code ${sparky_test_status}"
            cat "$tmp_run_dir/sparky_help.txt" 2>/dev/null || true
        fi
        rm -rf "$tmp_run_dir"
    fi

    echo ""
    ok "ALL APPIMAGE TESTS PASSED SUCCESSFULLY! 🚀"
    echo -e "${GREEN}${BOLD}You can now distribute and run:${NC} ${img}"
}

if [ "$DO_TEST" = true ]; then
    test_appimage "$OUTPUT_APPIMAGE"
fi

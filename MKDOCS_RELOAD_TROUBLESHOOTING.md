# MkDocs Auto-Reload Troubleshooting Guide

## Issue Description
When running `mkdocs serve` locally, changes to markdown files do not automatically refresh the browser. This requires manually stopping the server, restarting `mkdocs serve`, and refreshing the browser page.

## Configuration Changes Made

### Added Explicit Watch Paths
Added the following configuration to `mkdocs.yml`:

```yaml
watch:
  - docs
  - mkdocs.yml
```

This explicitly tells MkDocs to monitor:
1. The `docs/` directory (and all subdirectories) for changes
2. The `mkdocs.yml` configuration file itself

## How It Works
- MkDocs uses a file watcher to detect changes in specified directories
- When changes are detected, it rebuilds the affected pages
- The built-in live reload server sends a signal to the browser to refresh
- The `watch` configuration ensures MkDocs monitors all relevant files including:
  - Markdown files (`.md`)
  - CSS files (`docs/stylesheets/extra.css`)
  - Assets (images, fonts, etc.)
  - Configuration files

## Additional Local Troubleshooting Steps

If the issue persists after this configuration change, try the following:

### 1. Browser-Related Issues
- **Hard refresh**: Try `Ctrl+F5` (Windows/Linux) or `Cmd+Shift+R` (Mac) to force a hard refresh
- **Disable browser cache**: Open Developer Tools (F12) and enable "Disable cache" in the Network tab
- **Try a different browser**: Some browsers may cache more aggressively
- **Clear browser cache**: Clear your browser's cache and cookies

### 2. Port and Network Issues
- **Check if the port is blocked**: The default port is 8000. Try using a different port:
  ```bash
  mkdocs serve -a localhost:8001
  ```
- **Firewall/Antivirus**: Check if your firewall or antivirus is blocking WebSocket connections needed for live reload

### 3. File System Watchers
- **Linux (inotify limits)**: If you're on Linux and have many files, you might hit inotify limits:
  ```bash
  # Check current limit
  cat /proc/sys/fs/inotify/max_user_watches
  
  # Increase limit temporarily
  sudo sysctl fs.inotify.max_user_watches=524288
  
  # Make it permanent (add to /etc/sysctl.conf)
  echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
  sudo sysctl -p
  ```

- **Windows**: Ensure Windows Defender or antivirus isn't blocking file access

- **Network drives/Cloud storage**: Avoid running MkDocs from network drives, Dropbox, OneDrive, etc., as they can interfere with file watchers

### 4. Virtual Environment Issues
- **Reinstall MkDocs and dependencies**:
  ```bash
  pip uninstall mkdocs mkdocs-material mkdocstrings
  pip install -e ".[docu]"
  ```

### 5. MkDocs Process Issues
- **Kill zombie processes**: Check if old MkDocs processes are still running:
  ```bash
  # Linux/Mac
  ps aux | grep mkdocs
  kill -9 <PID>
  
  # Windows
  tasklist | findstr mkdocs
  taskkill /F /PID <PID>
  ```

### 6. Plugin-Related Issues
The `mkdocstrings` plugin can sometimes cause watch issues. Try:
- Temporarily disabling plugins one by one to identify the culprit
- Updating all plugins to the latest versions:
  ```bash
  pip install --upgrade mkdocs mkdocs-material mkdocstrings mkdocstrings-python
  ```

### 7. Use Verbose Mode
Run MkDocs with verbose output to see what it's watching:
```bash
mkdocs serve --verbose
```

This will show you:
- What files are being watched
- When rebuilds are triggered
- Any errors in the build process

### 8. Check for Symbolic Links
Symbolic links in the docs directory can sometimes cause issues with file watchers. If you have symlinks, try:
- Replacing them with actual files/directories
- Using the `--watch-theme` flag:
  ```bash
  mkdocs serve --watch-theme
  ```

### 9. System-Specific Issues

**macOS:**
- Update to the latest Python and pip versions
- Some versions have issues with the fsevents library

**WSL (Windows Subsystem for Linux):**
- File watchers may not work properly with files on Windows drives (e.g., `/mnt/c/`)
- Move your project to the WSL filesystem (e.g., `~/projects/`)

**Docker/Containers:**
- File watching may not work when mounting volumes
- Use polling mode if available (though this is slower)

## Verifying the Fix

After applying the configuration changes:

1. Start the MkDocs server:
   ```bash
   mkdocs serve
   ```

2. Open your browser to `http://localhost:8000`

3. Edit any markdown file in the `docs/` directory

4. Save the file and watch the terminal - you should see:
   ```
   INFO    -  Detected change in '<filename>'
   INFO    -  Building documentation...
   ```

5. The browser should automatically refresh with your changes

## Technical Details

### Default Watch Behavior
By default, MkDocs watches:
- The `docs_dir` (default: `docs/`)
- The `mkdocs.yml` configuration file
- Theme files (if using a local theme)

### Why Explicit Watch Paths Help
Adding explicit `watch` paths ensures:
1. **Clarity**: Makes it clear what should be monitored
2. **Reliability**: Overcomes potential issues with default watching
3. **Custom files**: Ensures CSS, JavaScript, and asset files are monitored
4. **Plugin compatibility**: Some plugins may interfere with default watching

### Live Reload Architecture
- MkDocs uses a WebSocket connection for live reload
- The browser maintains a connection to the MkDocs server
- When files change, the server sends a reload signal through the WebSocket
- The browser receives the signal and refreshes the page

## References
- [MkDocs Configuration - Watch](https://www.mkdocs.org/user-guide/configuration/#watch)
- [MkDocs Serve Command](https://www.mkdocs.org/user-guide/cli/#mkdocs-serve)
- [Material for MkDocs - Live Reload](https://squidfunk.github.io/mkdocs-material/creating-your-site/#previewing-as-you-write)

## Still Having Issues?

If none of these solutions work, please:
1. Check the MkDocs GitHub issues: https://github.com/mkdocs/mkdocs/issues
2. Check the Material for MkDocs issues: https://github.com/squidfunk/mkdocs-material/issues
3. Provide the following information when seeking help:
   - Operating system and version
   - Python version (`python --version`)
   - MkDocs version (`mkdocs --version`)
   - Output of `mkdocs serve --verbose`
   - Browser and version
   - Any error messages in the terminal or browser console

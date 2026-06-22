import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class DashboardBuildHook(BuildHookInterface):
    """Rebuild the dashboard SPA into pixeltable_cli/server/static before the wheel is packaged.

    The `pxt` daemon serves this prebuilt bundle, but the build output is gitignored and the
    hatchling wheel build only *includes* whatever already sits in that directory (see the
    `artifacts` entry in pyproject.toml) - it never regenerates it. Without this hook a wheel
    can ship JavaScript that predates the API it talks to, which crashes the dashboard on load
    for every user of that release. Building here makes a stale bundle impossible regardless of
    who cuts the release.
    """

    PLUGIN_NAME = 'custom'

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        dashboard_dir = Path(self.root) / 'dashboard'
        # No dashboard sources present - eg building a wheel from the sdist, which excludes
        # dashboard/. Nothing to rebuild; fall back to whatever assets are already bundled.
        if not (dashboard_dir / 'package.json').exists():
            return
        if os.environ.get('PIXELTABLE_SKIP_DASHBOARD_BUILD'):
            self.app.display_warning('PIXELTABLE_SKIP_DASHBOARD_BUILD set; skipping dashboard build')
            return
        npm = shutil.which('npm')
        if npm is None:
            raise RuntimeError(
                'npm (Node.js >= 20) is required to build the dashboard SPA bundled in the wheel. '
                'Install Node.js, or set PIXELTABLE_SKIP_DASHBOARD_BUILD=1 to build without rebuilding it.'
            )
        self.app.display_info('Building dashboard SPA ...')
        subprocess.run([npm, 'install', '--silent'], cwd=dashboard_dir, check=True)
        subprocess.run([npm, 'run', 'build'], cwd=dashboard_dir, check=True)

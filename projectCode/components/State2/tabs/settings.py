#settingd tab'
from reactpy import component, html
import constants
@component
def SettingsTab(data=None):
    return html.div(
        {**constants.dashboardCard, "color": constants.UI["text_color"]},
        "Settings tab – coming soon.",
    )


# discover tab
from reactpy import component, html
import constants
@component
def DiscoverTab(data=None):
    return html.div(
        {**constants.dashboardCard, "color": constants.UI["text_color"]},
        "Discover tab – coming soon.",
    )

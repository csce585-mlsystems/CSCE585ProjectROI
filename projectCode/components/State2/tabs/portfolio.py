# portfolio tab
from reactpy import component, html
import constants
@component
def PortfolioTab(data=None):
    return html.div(
        {**constants.dashboardCard, "color": constants.UI["text_color"]},
        "Portfolio tab – coming soon.",
    )

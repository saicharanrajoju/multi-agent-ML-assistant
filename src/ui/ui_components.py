"""
Reusable design-system HTML component helpers.
Each function returns an HTML string.
Render with: st.markdown(html, unsafe_allow_html=True)
No business logic — presentation only.
"""


def card(title: str, body: str) -> str:
    return (
        f'<div class="ds-card">'
        f'<div class="ds-card-title">{title}</div>'
        f'{body}'
        f'</div>'
    )


def banner(message: str, kind: str = "info", title: str = "") -> str:
    icons = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✕"}
    icon = icons.get(kind, "ℹ")
    title_html = f'<span class="ds-banner-title">{title}</span>' if title else ""
    return (
        f'<div class="ds-banner ds-banner-{kind}">'
        f'<span class="ds-banner-icon">{icon}</span>'
        f'<div class="ds-banner-body">{title_html}{message}</div>'
        f'</div>'
    )


def kpi(label: str, value: str, pill: str = "", pill_color: str = "gray") -> str:
    pill_html = (
        f'<div class="ds-kpi-pill ds-kpi-pill-{pill_color}">{pill}</div>'
        if pill else ""
    )
    return (
        f'<div class="ds-kpi">'
        f'<div class="ds-kpi-label">{label}</div>'
        f'<div class="ds-kpi-value">{value}</div>'
        f'{pill_html}'
        f'</div>'
    )


def kpi_row(items: list) -> str:
    """items = list of dicts: {label, value, pill, pill_color}"""
    cols = "".join(
        f'<div style="flex:1;min-width:120px">{kpi(**item)}</div>'
        for item in items
    )
    return f'<div style="display:flex;gap:0.75rem;flex-wrap:wrap">{cols}</div>'


def step_tracker(steps: list, current: int, is_complete: bool) -> str:
    parts = []
    for i, step in enumerate(steps):
        if i < current:
            cls = "ds-step-done"
            icon = "✓"
        elif i == current and not is_complete:
            cls = "ds-step-active"
            icon = "●"
        else:
            cls = "ds-step-pending"
            icon = str(i + 1)
        label = step.get("label", step.get("name", ""))
        parts.append(
            f'<div class="ds-step-pill {cls}">'
            f'<span>{icon}</span>{label}'
            f'</div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="ds-step-connector"></div>')
    return f'<div class="ds-steps">{"".join(parts)}</div>'


def empty_state(icon: str, title: str, desc: str, hint: str = "") -> str:
    hint_html = f'<div class="ds-empty-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="ds-empty">'
        f'<div class="ds-empty-icon">{icon}</div>'
        f'<div class="ds-empty-title">{title}</div>'
        f'<div class="ds-empty-desc">{desc}</div>'
        f'{hint_html}'
        f'</div>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    sub = (
        f'<p style="margin:2px 0 0;font-size:.85rem;color:var(--text-sec)">{subtitle}</p>'
        if subtitle else ""
    )
    return f'<div style="margin-bottom:1rem"><h3 style="margin:0">{title}</h3>{sub}</div>'


def divider() -> str:
    return '<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">'


def pill(text: str, color: str = "gray") -> str:
    return f'<span class="ds-kpi-pill ds-kpi-pill-{color}">{text}</span>'

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from llm_service import LLMConfigurationError, LLMGenerationError, generate_protocol
from models import ExperimentProtocol


CATALOG_PATH = Path(__file__).with_name("catalog.csv")
CATALOG_COLUMNS = {"item_name", "unit_size", "price_usd"}
FUZZY_MATCH_THRESHOLD = 0.72


def normalize_text(value: str) -> str:
    """Normalize text to improve fuzzy comparison stability."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    """Load the reagent catalog with schema validation."""
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    try:
        dataframe = pd.read_csv(catalog_path)
    except Exception as exc:
        raise ValueError(f"Unable to read catalog.csv: {exc}") from exc

    missing_columns = CATALOG_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"catalog.csv is missing required columns: {missing}")

    dataframe = dataframe.copy()
    dataframe["item_name"] = dataframe["item_name"].astype(str).str.strip()
    dataframe["unit_size"] = dataframe["unit_size"].astype(str).str.strip()
    dataframe["price_usd"] = pd.to_numeric(dataframe["price_usd"], errors="coerce")

    if dataframe["item_name"].eq("").any():
        raise ValueError("catalog.csv contains blank item_name values.")
    if dataframe["price_usd"].isna().any():
        raise ValueError("catalog.csv contains non-numeric price_usd values.")

    return dataframe


def parse_quantity(quantity_needed: str) -> tuple[float | None, str | None]:
    """Extract a numeric quantity and unit from free-text reagent amounts."""
    match = re.search(r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)", quantity_needed.strip())
    if not match:
        return None, None

    amount = float(match.group("amount"))
    unit = match.group("unit").lower()
    return amount, unit


def parse_catalog_unit_size(unit_size: str) -> tuple[float | None, str | None]:
    """Parse catalog pack sizes such as '500 g' or '1 L'."""
    match = re.search(r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)", unit_size.strip())
    if not match:
        return None, None

    amount = float(match.group("amount"))
    unit = match.group("unit").lower()
    return amount, unit


def convert_to_base_unit(amount: float, unit: str) -> tuple[float, str] | tuple[None, None]:
    """Convert compatible units to shared bases for cost estimation."""
    conversions = {
        "mg": (0.001, "g"),
        "g": (1.0, "g"),
        "kg": (1000.0, "g"),
        "ul": (0.001, "ml"),
        "ml": (1.0, "ml"),
        "l": (1000.0, "ml"),
    }
    if unit not in conversions:
        return None, None

    multiplier, base_unit = conversions[unit]
    return amount * multiplier, base_unit


def match_catalog_item(reagent_name: str, catalog: pd.DataFrame) -> tuple[pd.Series | None, float]:
    """Find the best catalog row using exact matching first, then fuzzy similarity."""
    target = normalize_text(reagent_name)
    best_row: pd.Series | None = None
    best_score = 0.0

    for _, row in catalog.iterrows():
        candidate = normalize_text(str(row["item_name"]))
        score = 1.0 if candidate == target else SequenceMatcher(None, target, candidate).ratio()
        if score > best_score:
            best_row = row
            best_score = score

    if best_row is None or best_score < FUZZY_MATCH_THRESHOLD:
        return None, best_score
    return best_row, best_score


def estimate_reagent_costs(protocol: ExperimentProtocol, catalog: pd.DataFrame) -> pd.DataFrame:
    """Match protocol reagents to the catalog and estimate costs where possible."""
    rows: list[dict[str, Any]] = []

    for reagent in protocol.required_reagents:
        catalog_row, score = match_catalog_item(reagent.item_name, catalog)
        requested_amount, requested_unit = parse_quantity(reagent.quantity_needed)

        price_display: str | float = "Price unknown"
        matched_item = "Not found"
        pack_size = "-"

        if catalog_row is not None:
            matched_item = str(catalog_row["item_name"])
            pack_size = str(catalog_row["unit_size"])

            pack_amount, pack_unit = parse_catalog_unit_size(pack_size)
            if None not in (requested_amount, requested_unit, pack_amount, pack_unit):
                requested_base_amount, requested_base_unit = convert_to_base_unit(
                    requested_amount,
                    requested_unit,
                )
                pack_base_amount, pack_base_unit = convert_to_base_unit(pack_amount, pack_unit)

                if (
                    requested_base_amount is not None
                    and pack_base_amount is not None
                    and requested_base_unit == pack_base_unit
                ):
                    units_to_buy = max(1, math.ceil(requested_base_amount / pack_base_amount))
                    price_display = round(units_to_buy * float(catalog_row["price_usd"]), 2)

        rows.append(
            {
                "Reagent": reagent.item_name,
                "Quantity Needed": reagent.quantity_needed,
                "Purpose": reagent.purpose,
                "Matched Catalog Item": matched_item,
                "Catalog Unit Size": pack_size,
                "Match Score": round(score, 2) if catalog_row is not None else "-",
                "Estimated Cost (USD)": price_display,
            }
        )

    return pd.DataFrame(rows)


def calculate_total_budget(reagent_costs: pd.DataFrame) -> float:
    """Sum only known reagent prices into a total estimated budget."""
    numeric_costs = pd.to_numeric(reagent_costs["Estimated Cost (USD)"], errors="coerce").fillna(0.0)
    return round(float(numeric_costs.sum()), 2)


def render_header() -> None:
    """Render the app title and user-facing description."""
    st.title("AI Scientist OS")
    st.markdown(
        """
        Generate a structured laboratory protocol from a scientific hypothesis,
        then estimate reagent spend by matching required materials against a local catalog.
        """
    )


def render_protocol(protocol: ExperimentProtocol) -> None:
    """Render the validated protocol with a professional Streamlit layout."""
    st.subheader(protocol.experiment_title)

    metric_col, time_col = st.columns(2)
    metric_col.metric("Feasibility Score", f"{protocol.feasibility_score}/10")
    time_col.metric("Estimated Time", f"{protocol.estimated_time_hours:.1f} hours")

    st.markdown("**Safety Hazards**")
    for hazard in protocol.safety_hazards:
        st.warning(hazard, icon="⚠️")

    steps_df = pd.DataFrame(
        [
            {"Step Number": step.step_number, "Action": step.action}
            for step in protocol.chronological_steps
        ]
    )

    st.markdown("**Chronological Steps**")
    st.dataframe(steps_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="AI Scientist OS",
        page_icon=":test_tube:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    render_header()

    try:
        catalog = load_catalog(str(CATALOG_PATH))
    except Exception as exc:
        st.error(f"Failed to load catalog.csv: {exc}")
        st.stop()

    with st.container(border=True):
        hypothesis = st.text_area(
            "Scientific Hypothesis",
            height=180,
            placeholder=(
                "Example: Increasing extracellular glucose concentration will improve "
                "short-term ATP production in cultured mammalian cells."
            ),
        )
        generate_clicked = st.button("Generate Protocol", type="primary", use_container_width=True)

    if not generate_clicked:
        return

    if not hypothesis.strip():
        st.warning("Enter a scientific hypothesis before generating a protocol.")
        st.stop()

    try:
        with st.spinner("Generating laboratory protocol and estimating reagent budget..."):
            protocol_payload = generate_protocol(hypothesis)
            protocol = ExperimentProtocol.model_validate(protocol_payload)
            reagent_costs = estimate_reagent_costs(protocol, catalog)
            total_budget = calculate_total_budget(reagent_costs)
    except LLMConfigurationError as exc:
        st.error(str(exc))
        st.stop()
    except LLMGenerationError as exc:
        st.error(str(exc))
        st.stop()
    except ValidationError as exc:
        st.error(f"Protocol validation error: {exc}")
        st.stop()
    except Exception as exc:  # pragma: no cover - final UI safety net
        st.error(f"Unexpected application error: {exc}")
        st.stop()

    render_protocol(protocol)

    st.markdown("**Required Reagents and Cost Estimate**")
    st.dataframe(reagent_costs, use_container_width=True, hide_index=True)
    st.metric("Total Estimated Budget", f"${total_budget:,.2f}")


if __name__ == "__main__":
    main()

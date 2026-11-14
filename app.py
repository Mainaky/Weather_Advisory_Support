"""
app.py - Streamlit Frontend for Weather-Advisory Support Bot

Features:
1. Conversational Chat Tab (backed by LangGraph agent & Open-Meteo verified live data).
   - Real-time forecast evaluation.
   - Explanatory 'Why this answer?' expander:
     * Primary SOP ID & Title
     * Primary Selection Rationale (explaining why this rule led)
     * Secondary matched policies with full advice text
     * Verified weather metrics from Open-Meteo
2. SOP Policy Manager Tab (Live In-UI Policy Editor):
   - Zero-code policy updates directly from the web browser.
   - View, search, and filter all active policies.
   - Interactive form to edit any existing SOP and write changes directly to sops.yaml.
   - Interactive form to add a brand-new SOP on the spot (for live interview testing).
   - Direct raw YAML editor for advanced bulk adjustments.
3. Multi-turn Session Memory backed by LangGraph MemorySaver.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import yaml

import graph
import weather

# Page configuration
st.set_page_config(
    page_title="Weather-Advisory Support Bot",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOPS_FILE = Path(__file__).resolve().parent / "sops.yaml"

# Session state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


def reset_session():
    """Resets memory checkpointer thread and clears conversation history."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()


def save_sops_to_disk(sops_list: List[Dict[str, Any]]) -> bool:
    """Safely saves a list of SOP dictionaries back to sops.yaml."""
    try:
        with open(SOPS_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(sops_list, f, sort_keys=False, allow_unicode=True)
        return True
    except Exception as e:
        st.error(f"Error saving sops.yaml: {e}")
        return False


# ==============================================================================
# SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("⛅ Advisory Controls")
    st.caption("Backed by LangGraph Agent & Verified Open-Meteo Data")

    st.markdown(f"**Session Thread ID:** `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 Start New Chat Session", use_container_width=True):
        reset_session()

    st.markdown("---")
    st.markdown("### 📋 System Health")
    current_sops = graph.load_all_sops()
    st.metric("Active Policies in sops.yaml", len(current_sops))
    
    # Provider & model info
    model_name = os.getenv("MODEL", "gemini-3.5-flash-lite")
    st.caption(f"Active LLM: `{model_name}`")

    st.markdown("---")
    st.markdown("### 💡 Quick Starters")
    starter_prompts = [
        "Is it safe to cycle in Bhopal today?",
        "Is this the perfect weather traveling in Bhopal?",
        "Is today a good day for a picnic in London?",
        "Should my grandmother go for an evening walk in Delhi?",
        "Thinking of riding my scooter to college in Mumbai, any worries?",
        "Is it safe to go scuba diving in the Andamans?",
    ]
    for prompt_text in starter_prompts:
        if st.button(prompt_text, key=f"quick_{prompt_text}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()


# ==============================================================================
# MAIN TABS: CHAT & SOP POLICY MANAGER
# ==============================================================================
st.title("🛡️ Weather-Advisory Support Bot")
st.markdown(
    "Outdoor activity safety advisor grounded strictly in **live Open-Meteo forecasts** and governed by "
    "**approved Standard Operating Procedures (SOPs)**. Never invents advice; citations and numbers are 100% traceable."
)

tab_chat, tab_manager = st.tabs(["💬 Weather Advisory Chat", "⚙️ SOP Policy Manager (Live UI Editor)"])


# ------------------------------------------------------------------------------
# TAB 1: ADVISORY CHAT
# ------------------------------------------------------------------------------
with tab_chat:
    # Display conversation messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            metadata = msg.get("metadata")
            if metadata and msg["role"] == "assistant":
                with st.expander("🔍 Why this answer? (Traceability & Policy Audit)"):
                    chosen_sop = metadata.get("chosen_sop")
                    selection_reason = metadata.get("selection_reason")
                    other_sops = metadata.get("other_sops") or []
                    facts = metadata.get("weather_facts")

                    err_msg = metadata.get("error")
                    if chosen_sop:
                        sev = chosen_sop.get("severity", 1)
                        sev_badge = {
                            1: "🟢 Severity 1 (Informational Clearance)",
                            2: "🟡 Severity 2 (Caution / Advisory)",
                            3: "🟠 Severity 3 (Warning / Hazard)",
                            4: "🔴 Severity 4 (Emergency / Danger)",
                        }.get(sev, f"Severity {sev}")

                        st.markdown(f"**🎯 Primary Policy Applied:** `{chosen_sop.get('id')}` — **{chosen_sop.get('title')}**")
                        st.markdown(f"**Category:** `{chosen_sop.get('category')}` | **Hazard Rating:** {sev_badge}")
                        st.markdown(f"**Rule Rationale:** {chosen_sop.get('rationale')}")

                        if selection_reason:
                            st.info(f"**🧠 Primary Selection Rationale:** {selection_reason}")

                        if other_sops:
                            st.markdown("---")
                            st.markdown(f"**📑 Secondary Policies Triggered ({len(other_sops)}):**")
                            for osop in other_sops:
                                o_sev = osop.get("severity", 1)
                                o_badge = {1: "🟢 Info", 2: "🟡 Caution", 3: "🟠 Warning", 4: "🔴 Danger"}.get(o_sev, f"Sev {o_sev}")
                                st.markdown(f"- `{osop.get('id')}`: **{osop.get('title')}** ({o_badge})")
                                if osop.get("advice"):
                                    st.caption(f"  *Approved text: {osop.get('advice')[:120]}...*")
                    elif err_msg:
                        st.info(f"ℹ️ **Operational Notice:** {err_msg}")
                    else:
                        st.markdown("*Out-of-scope activity or unmapped topic. Returned standard honest boundary response.*")

                    if facts and not err_msg:
                        st.markdown("---")
                        loc_title = facts.get("location_name") or "Queried Location"
                        st.markdown(f"**📊 Verified Weather Facts from Open-Meteo ({loc_title}):**")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Temperature", f"{facts.get('temperature_2m')} °C")
                            st.metric("Feels Like", f"{facts.get('apparent_temperature')} °C")
                            st.metric("UV Index", f"{facts.get('uv_index')}")
                        with col2:
                            st.metric("Wind Speed", f"{facts.get('wind_speed_10m')} km/h")
                            st.metric("Wind Gusts", f"{facts.get('wind_gusts_10m')} km/h")
                            st.metric("Weather Code", f"{facts.get('weather_code')}")
                        with col3:
                            st.metric("Precipitation Prob.", f"{facts.get('precipitation_probability')} %")
                            st.metric("Current Rain", f"{facts.get('precipitation')} mm")
                            st.metric("Daily Rain Total", f"{facts.get('daily_precipitation_sum')} mm")

    # Chat Input handler
    user_input = st.chat_input("Ask an outdoor safety question (e.g., 'Is it safe to cycle in Bhopal today?')...")
    if hasattr(st.session_state, "pending_prompt") and st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_input:
        # Append and show user query
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run LangGraph pipeline
        with st.chat_message("assistant"):
            with st.spinner("Fetching live forecast & evaluating SOP policies..."):
                result = graph.ask_bot(user_input, thread_id=st.session_state.thread_id)
                bot_reply = result.get("reply", "I don't have guidance for that.")
                chosen_sop = result.get("chosen_sop")
                selection_reason = result.get("selection_reason")
                other_sops = result.get("other_sops") or []
                facts = result.get("weather_facts")

                st.markdown(bot_reply)

                # Show real-time trace expander
                with st.expander("🔍 Why this answer? (Traceability & Policy Audit)"):
                    err_msg = metadata.get("error")
                    if chosen_sop:
                        sev = chosen_sop.get("severity", 1)
                        sev_badge = {
                            1: "🟢 Severity 1 (Informational Clearance)",
                            2: "🟡 Severity 2 (Caution / Advisory)",
                            3: "🟠 Severity 3 (Warning / Hazard)",
                            4: "🔴 Severity 4 (Emergency / Danger)",
                        }.get(sev, f"Severity {sev}")

                        st.markdown(f"**🎯 Primary Policy Applied:** `{chosen_sop.get('id')}` — **{chosen_sop.get('title')}**")
                        st.markdown(f"**Category:** `{chosen_sop.get('category')}` | **Hazard Rating:** {sev_badge}")
                        st.markdown(f"**Rule Rationale:** {chosen_sop.get('rationale')}")

                        if selection_reason:
                            st.info(f"**🧠 Primary Selection Rationale:** {selection_reason}")

                        if other_sops:
                            st.markdown("---")
                            st.markdown(f"**📑 Secondary Policies Triggered ({len(other_sops)}):**")
                            for osop in other_sops:
                                o_sev = osop.get("severity", 1)
                                o_badge = {1: "🟢 Info", 2: "🟡 Caution", 3: "🟠 Warning", 4: "🔴 Danger"}.get(o_sev, f"Sev {o_sev}")
                                st.markdown(f"- `{osop.get('id')}`: **{osop.get('title')}** ({o_badge})")
                                if osop.get("advice"):
                                    st.caption(f"  *Approved text: {osop.get('advice')[:120]}...*")
                    elif err_msg:
                        st.info(f"ℹ️ **Operational Notice:** {err_msg}")
                    else:
                        st.markdown("*Out-of-scope activity or unmapped topic. Returned standard honest boundary response.*")

                    if facts and not err_msg:
                        st.markdown("---")
                        loc_title = facts.get("location_name") or "Queried Location"
                        st.markdown(f"**📊 Verified Weather Facts from Open-Meteo ({loc_title}):**")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Temperature", f"{facts.get('temperature_2m')} °C")
                            st.metric("Feels Like", f"{facts.get('apparent_temperature')} °C")
                            st.metric("UV Index", f"{facts.get('uv_index')}")
                        with col2:
                            st.metric("Wind Speed", f"{facts.get('wind_speed_10m')} km/h")
                            st.metric("Wind Gusts", f"{facts.get('wind_gusts_10m')} km/h")
                            st.metric("Weather Code", f"{facts.get('weather_code')}")
                        with col3:
                            st.metric("Precipitation Prob.", f"{facts.get('precipitation_probability')} %")
                            st.metric("Current Rain", f"{facts.get('precipitation')} mm")
                            st.metric("Daily Rain Total", f"{facts.get('daily_precipitation_sum')} mm")

        # Save assistant turn
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_reply,
            "metadata": {
                "chosen_sop": chosen_sop,
                "selection_reason": selection_reason,
                "other_sops": other_sops,
                "weather_facts": facts if not result.get("error") else None,
                "error": result.get("error"),
            }
        })


# ------------------------------------------------------------------------------
# TAB 2: SOP POLICY MANAGER (LIVE IN-UI POLICY EDITOR)
# ------------------------------------------------------------------------------
with tab_manager:
    st.header("⚙️ Standard Operating Procedure (SOP) Manager")
    st.markdown(
        "Manage, edit, or add company safety policies dynamically without restarting or modifying control-flow code. "
        "Any change saved here updates `sops.yaml` on disk and is loaded fresh on the next user query."
    )

    all_sops = graph.load_all_sops()

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total SOPs", len(all_sops))
    with m2:
        categories = sorted(list(set(s.get("category", "other") for s in all_sops)))
        st.metric("Categories", len(categories))
    with m3:
        high_sev = sum(1 for s in all_sops if s.get("severity", 1) >= 3)
        st.metric("High Severity (3-4)", high_sev)
    with m4:
        fuzzy_count = sum(1 for s in all_sops if s.get("type") == "fuzzy")
        st.metric("Fuzzy Policies", fuzzy_count)

    st.markdown("---")

    # Sub-tabs for management
    sub_view, sub_edit, sub_add, sub_raw = st.tabs([
        "📖 View Active Policies",
        "✏️ Edit Existing Policy",
        "➕ Add New Policy (Live Interview Test)",
        "📄 Raw YAML Editor"
    ])

    # 1. VIEW ACTIVE POLICIES
    with sub_view:
        st.subheader("Active Policy Manual")
        
        # Search & filters
        c_filter1, c_filter2 = st.columns([2, 1])
        with c_filter1:
            search_term = st.text_input("Search policies by ID, title, or keyword:", placeholder="e.g. wind, picnic, travel, rain")
        with c_filter2:
            cat_choice = st.selectbox("Filter by Category:", ["All"] + categories)

        filtered_sops = all_sops
        if cat_choice != "All":
            filtered_sops = [s for s in filtered_sops if s.get("category") == cat_choice]
        if search_term:
            term = search_term.lower()
            filtered_sops = [
                s for s in filtered_sops
                if term in s.get("id", "").lower()
                or term in s.get("title", "").lower()
                or term in s.get("rationale", "").lower()
                or any(term in tag.lower() for tag in s.get("applies_to", []))
            ]

        st.caption(f"Showing {len(filtered_sops)} of {len(all_sops)} policies")

        for s in filtered_sops:
            sev = s.get("severity", 1)
            sev_badge = {1: "🟢 Info (1)", 2: "🟡 Caution (2)", 3: "🟠 Warning (3)", 4: "🔴 Danger (4)"}.get(sev, f"Sev {sev}")
            with st.expander(f"**{s.get('id')}**: {s.get('title')} — {sev_badge} [{s.get('category')}]"):
                st.markdown(f"**Applies To:** `{', '.join(s.get('applies_to', []))}`")
                st.markdown(f"**Type:** `{s.get('type')}`")
                if s.get("type") == "numeric":
                    st.code(s.get("condition", ""), language="python")
                else:
                    st.markdown(f"**Fuzzy Description:** {s.get('fuzzy_description')}")
                    st.markdown("**Soft Conditions:**")
                    for sc in s.get("soft_conditions", []):
                        st.markdown(f"- `{sc}`")

                st.markdown(f"**Approved Advice Template:**\n> {s.get('advice')}")
                st.markdown(f"**Rationale:** *{s.get('rationale')}*")

    # 2. EDIT EXISTING POLICY
    with sub_edit:
        st.subheader("Edit an Existing Policy")
        st.markdown("Select a policy from the dropdown below to modify its conditions, advice, or severity.")

        sop_ids = [s.get("id") for s in all_sops]
        selected_id = st.selectbox("Select SOP to edit:", sop_ids)
        target_sop = next((s for s in all_sops if s.get("id") == selected_id), None)

        if target_sop:
            with st.form(key=f"edit_form_{selected_id}"):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    e_title = st.text_input("Title", value=target_sop.get("title", ""))
                    e_category = st.selectbox(
                        "Category",
                        ["outdoor_exercise", "travel", "vulnerable_groups", "general_outdoor", "other"],
                        index=["outdoor_exercise", "travel", "vulnerable_groups", "general_outdoor", "other"].index(
                            target_sop.get("category", "outdoor_exercise") if target_sop.get("category") in ["outdoor_exercise", "travel", "vulnerable_groups", "general_outdoor"] else "other"
                        )
                    )
                    e_type = st.selectbox("Rule Type", ["numeric", "fuzzy"], index=0 if target_sop.get("type") == "numeric" else 1)
                with col_e2:
                    e_severity = st.slider("Hazard Severity Rating", min_value=1, max_value=4, value=int(target_sop.get("severity", 1)))
                    e_applies = st.text_input("Applies To (comma-separated tags)", value=", ".join(target_sop.get("applies_to", [])))

                if e_type == "numeric":
                    e_condition = st.text_input("AST Condition Expression", value=target_sop.get("condition", ""))
                else:
                    e_condition = None
                    e_fuzzy_desc = st.text_input("Fuzzy Scenario Description", value=target_sop.get("fuzzy_description", ""))
                    e_soft_conds = st.text_area("Soft Conditions (one per line)", value="\n".join(target_sop.get("soft_conditions", [])))

                e_advice = st.text_area("Approved Advice Template (use {placeholders} for verified weather values)", value=target_sop.get("advice", ""), height=100)
                st.caption("Available placeholders: `{temperature_2m}`, `{apparent_temperature}`, `{wind_speed_10m}`, `{wind_gusts_10m}`, `{precipitation}`, `{precipitation_probability}`, `{uv_index}`, `{location_name}`, `{activity}`")

                e_rationale = st.text_input("Rule Rationale", value=target_sop.get("rationale", ""))

                submitted = st.form_submit_button("💾 Save Changes to sops.yaml", use_container_width=True)
                if submitted:
                    # Update target in-place
                    target_sop["title"] = e_title.strip()
                    target_sop["category"] = e_category.strip()
                    target_sop["severity"] = int(e_severity)
                    target_sop["type"] = e_type
                    target_sop["applies_to"] = [tag.strip() for tag in e_applies.split(",") if tag.strip()]
                    target_sop["advice"] = e_advice.strip()
                    target_sop["rationale"] = e_rationale.strip()

                    if e_type == "numeric":
                        target_sop["condition"] = e_condition.strip() if e_condition else ""
                    else:
                        target_sop["fuzzy_description"] = e_fuzzy_desc.strip()
                        target_sop["soft_conditions"] = [line.strip() for line in e_soft_conds.split("\n") if line.strip()]

                    if save_sops_to_disk(all_sops):
                        st.success(f"✅ Successfully updated **{selected_id}**! Changes are live on disk and ready for immediate testing.")
                        st.rerun()

    # 3. ADD NEW POLICY (LIVE INTERVIEW DEMO)
    with sub_add:
        st.subheader("➕ Add a New Policy on the Spot")
        st.markdown(
            "Designed specifically for the live interview test: add an 11th / 17th SOP live without modifying any control-flow code! "
            "Once saved, the bot dynamically detects and enforces the new rule instantly."
        )

        with st.form(key="add_new_sop_form"):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                next_id = f"SOP-{len(all_sops) + 1:03d}"
                new_id = st.text_input("Policy ID", value=next_id)
                new_title = st.text_input("Policy Title", placeholder="e.g. Kayaking and Water Sports Surface Wind Advisory")
                new_category = st.selectbox("Category", ["outdoor_exercise", "travel", "vulnerable_groups", "general_outdoor"])
            with col_a2:
                new_severity = st.slider("Severity (1=Info, 2=Caution, 3=Warning, 4=Danger)", min_value=1, max_value=4, value=2)
                new_type = st.selectbox("Policy Type", ["numeric", "fuzzy"])
                new_applies = st.text_input("Applies To Tags (comma-separated)", placeholder="e.g. kayaking, boating, water_sports")

            if new_type == "numeric":
                new_condition = st.text_input(
                    "Condition Expression",
                    placeholder="e.g. wind_speed_10m >= 25.0 or wind_gusts_10m >= 35.0"
                )
            else:
                new_condition = ""
                new_fuzzy_desc = st.text_input("Fuzzy Scenario Description", placeholder="e.g. user planning outdoor water activities")
                new_soft_conds = st.text_area("Soft Conditions (one per line)", placeholder="wind_speed_10m < 20.0\nprecipitation < 1.0")

            new_advice = st.text_area(
                "Approved Advice Template",
                placeholder="High surface winds ({wind_speed_10m} km/h) make water sports unsafe in {location_name}. Policy: " + next_id + ".",
                height=90
            )
            new_rationale = st.text_input("Rule Rationale", placeholder="e.g. Elevated open-water chop poses capsize and safety risks.")

            add_btn = st.form_submit_button("🚀 Add New Policy & Save to Disk", use_container_width=True)
            if add_btn:
                # Validation
                if any(s.get("id") == new_id.strip() for s in all_sops):
                    st.error(f"Policy ID '{new_id}' already exists! Please choose a unique ID.")
                elif not new_title.strip() or not new_advice.strip():
                    st.error("Please provide both a Title and an Approved Advice Template.")
                else:
                    new_entry: Dict[str, Any] = {
                        "id": new_id.strip(),
                        "title": new_title.strip(),
                        "category": new_category,
                        "severity": int(new_severity),
                        "type": new_type,
                        "applies_to": [t.strip() for t in new_applies.split(",") if t.strip()],
                        "advice": new_advice.strip(),
                        "rationale": new_rationale.strip(),
                    }
                    if new_type == "numeric":
                        new_entry["condition"] = new_condition.strip()
                    else:
                        new_entry["fuzzy_description"] = new_fuzzy_desc.strip()
                        new_entry["soft_conditions"] = [line.strip() for line in new_soft_conds.split("\n") if line.strip()]

                    all_sops.append(new_entry)
                    if save_sops_to_disk(all_sops):
                        st.success(f"🎉 New policy **{new_id}** successfully created and saved to `sops.yaml`! It is active immediately.")
                        st.rerun()

    # 4. RAW YAML EDITOR
    with sub_raw:
        st.subheader("Direct YAML File Editor")
        st.markdown("Directly inspect or edit the full `sops.yaml` policy file.")
        
        try:
            with open(SOPS_FILE, "r", encoding="utf-8") as f:
                raw_yaml_text = f.read()
        except Exception:
            raw_yaml_text = ""

        edited_yaml = st.text_area("sops.yaml content:", value=raw_yaml_text, height=450)
        if st.button("💾 Save Direct YAML Changes", use_container_width=True):
            try:
                parsed = yaml.safe_load(edited_yaml)
                if not isinstance(parsed, list):
                    st.error("Invalid YAML structure: Must be a list of SOP policy dictionaries.")
                else:
                    with open(SOPS_FILE, "w", encoding="utf-8") as f:
                        f.write(edited_yaml)
                    st.success("✅ `sops.yaml` updated successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"YAML Syntax Error: {e}")

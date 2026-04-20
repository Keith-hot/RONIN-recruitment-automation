"""
RONIN Digital Research – ROI Analysis & Visualization Dashboard
Professional B2B research recruitment metrics with interactive Plotly charts.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src/ is on path so we can import the lead scorer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lead_scoring import LeadScoringModel


# ------------------------------------------------------------------
# Constants & Assumptions
# ------------------------------------------------------------------
HOURLY_RATE_USD = 25.0
MINUTES_PER_LEAD_MANUAL = 15.0
SETUP_COST_USD = 2_500.0
REVIEW_RATE = 0.20
DEAL_VALUE_USD = 500.0
LIFT_MULTIPLIER = 3.2
WORKING_DAYS_PER_MONTH = 22.0


# ------------------------------------------------------------------
# ROI Calculator
# ------------------------------------------------------------------
class ROICalculator:
    """Computes labour-cost savings, revenue uplift, and break-even metrics."""

    @staticmethod
    def calculate_manual_cost(total_leads: int, hourly_rate: float, minutes_per_lead: float) -> float:
        """Total labour cost under fully manual outreach."""
        hours = total_leads * (minutes_per_lead / 60.0)
        return hours * hourly_rate

    @staticmethod
    def calculate_automation_cost(total_leads: int, setup_cost: float, review_rate: float) -> float:
        """Automation cost = fixed setup + human review for flagged leads only."""
        review_cost = total_leads * review_rate * (MINUTES_PER_LEAD_MANUAL / 60.0) * HOURLY_RATE_USD
        return setup_cost + review_cost

    @staticmethod
    def project_annual_savings(
        monthly_projects: int,
        avg_leads_per_project: int,
        deal_value: float,
        manual_participation_rate: float,
        ai_participation_rate: float,
    ) -> Dict[str, float]:
        """
        Projects annual labour savings and revenue uplift.
        Returns dictionary with break_even_days, annual_savings_usd, etc.
        """
        total_leads = monthly_projects * avg_leads_per_project

        # Labour costs
        manual_cost_monthly = ROICalculator.calculate_manual_cost(
            total_leads, HOURLY_RATE_USD, MINUTES_PER_LEAD_MANUAL
        )
        auto_cost_monthly = ROICalculator.calculate_automation_cost(
            total_leads, SETUP_COST_USD, REVIEW_RATE
        )
        monthly_labour_saving = manual_cost_monthly - auto_cost_monthly

        # Revenue
        manual_monthly_revenue = total_leads * manual_participation_rate * deal_value
        ai_monthly_revenue = total_leads * ai_participation_rate * deal_value
        monthly_revenue_uplift = ai_monthly_revenue - manual_monthly_revenue

        # Break-even (setup cost recovered by monthly savings)
        net_monthly_gain = monthly_labour_saving + monthly_revenue_uplift
        if net_monthly_gain > 0:
            break_even_days = (SETUP_COST_USD / net_monthly_gain) * 30.0
        else:
            break_even_days = 999.0

        annual_savings = monthly_labour_saving * 12.0
        annual_uplift = monthly_revenue_uplift * 12.0

        efficiency_gain_pct = (
            (manual_cost_monthly - auto_cost_monthly) / manual_cost_monthly * 100.0
            if manual_cost_monthly > 0 else 0.0
        )

        roi_pct = (
            (net_monthly_gain * 12.0) / SETUP_COST_USD * 100.0
            if SETUP_COST_USD > 0 else 0.0
        )

        return {
            "manual_monthly_cost": round(manual_cost_monthly, 2),
            "automation_monthly_cost": round(auto_cost_monthly, 2),
            "savings_percent": round(efficiency_gain_pct, 1),
            "break_even_days": round(break_even_days, 1),
            "annual_labour_savings_usd": round(annual_savings, 2),
            "annual_revenue_uplift_usd": round(annual_uplift, 2),
            "annual_total_benefit_usd": round(annual_savings + annual_uplift, 2),
            "roi_percentage": round(roi_pct, 1),
            "efficiency_gain_percent": round(efficiency_gain_pct, 1),
        }


# ------------------------------------------------------------------
# Dashboard Generator
# ------------------------------------------------------------------
class DashboardGenerator:
    """Produces four interactive HTML charts for the RONIN ROI dashboard."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Chart 1: Funnel Comparison
    # ------------------------------------------------------------------
    def build_funnel_comparison(
        self,
        manual_counts: List[int],
        ai_counts: List[int],
        stages: List[str],
    ) -> None:
        """Side-by-side funnel: Traditional Manual vs AI-Powered."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Traditional Manual Process", "AI-Powered Automation"),
            specs=[[{"type": "funnel"}, {"type": "funnel"}]],
        )

        # Manual – gray tones
        fig.add_trace(
            go.Funnel(
                y=stages,
                x=manual_counts,
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.75,
                marker={
                    "color": ["#718096", "#A0AEC0", "#CBD5E0", "#E2E8F0", "#EDF2F7"]
                },
                connector={"line": {"color": "#718096", "dash": "dot"}},
                name="Manual",
            ),
            row=1,
            col=1,
        )

        # AI – blue gradient
        fig.add_trace(
            go.Funnel(
                y=stages,
                x=ai_counts,
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.85,
                marker={
                    "color": ["#1a365d", "#2c5282", "#2b6cb0", "#4299e1", "#63b3ed"]
                },
                connector={"line": {"color": "#2b6cb0", "dash": "solid"}},
                name="AI-Powered",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text="<b>Recruitment Funnel Comparison</b><br><sup>Based on RONIN Digital Research workflow simulation</sup>",
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=13),
            height=600,
            width=1100,
            showlegend=False,
            template="plotly_white",
        )

        out_path = os.path.join(self.output_dir, "funnel_comparison.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"  Saved funnel comparison to {out_path}")

    # ------------------------------------------------------------------
    # Chart 2: ROI Timeline
    # ------------------------------------------------------------------
    def build_roi_timeline(
        self,
        daily_labor_saving: float,
        daily_revenue_uplift: float,
        break_even_day: float,
    ) -> None:
        """30-day cumulative ROI trajectory."""
        days = list(range(1, 31))
        labor = [daily_labor_saving * d for d in days]
        revenue = [daily_revenue_uplift * d for d in days]
        net = [labor[d - 1] + revenue[d - 1] - SETUP_COST_USD for d in days]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=days,
                y=labor,
                mode="lines+markers",
                name="Labour Savings",
                line=dict(color="#E53E3E", width=3),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=revenue,
                mode="lines+markers",
                name="Revenue Uplift",
                line=dict(color="#38A169", width=3),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=net,
                mode="lines+markers",
                name="Net ROI",
                line=dict(color="#3182CE", width=4),
                marker=dict(size=8),
            )
        )

        # Break-even annotation
        be_day = max(1, int(break_even_day))
        be_value = net[be_day - 1] if be_day <= 30 else net[-1]
        fig.add_vline(
            x=break_even_day,
            line=dict(color="#805AD5", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=break_even_day,
            y=be_value,
            text=f"<b>Break-even<br>Day {break_even_day:.0f}</b>",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-60,
            font=dict(size=12, color="#805AD5"),
        )

        fig.update_layout(
            title="<b>30-Day ROI Trajectory: Manual vs. AI-Powered Recruitment</b>",
            xaxis_title="Day",
            yaxis_title="Cumulative Value (USD)",
            font=dict(family="Arial, sans-serif", size=13),
            height=550,
            width=950,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )

        out_path = os.path.join(self.output_dir, "roi_timeline.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"  Saved ROI timeline to {out_path}")

    # ------------------------------------------------------------------
    # Chart 3: Lead Score Distribution
    # ------------------------------------------------------------------
    def build_lead_score_distribution(self, scores: List[float]) -> None:
        """Histogram of lead scores with colour-coded intent segments."""
        scores_arr = np.array(scores)

        fig = go.Figure()

        # Hard-to-reach / low intent (< 0.3)
        low = scores_arr[scores_arr < 0.3]
        fig.add_trace(
            go.Histogram(
                x=low,
                nbinsx=20,
                name="Hard-to-Reach (< 0.3)",
                marker_color="#FC8181",
                opacity=0.85,
            )
        )

        # Normal (0.3 – 0.8)
        mid = scores_arr[(scores_arr >= 0.3) & (scores_arr <= 0.8)]
        fig.add_trace(
            go.Histogram(
                x=mid,
                nbinsx=20,
                name="Normal (0.3 – 0.8)",
                marker_color="#F6E05E",
                opacity=0.85,
            )
        )

        # High-intent (> 0.8)
        high = scores_arr[scores_arr > 0.8]
        fig.add_trace(
            go.Histogram(
                x=high,
                nbinsx=20,
                name="High-Intent (> 0.8)",
                marker_color="#68D391",
                opacity=0.85,
            )
        )

        fig.add_vline(
            x=0.8,
            line=dict(color="#2D3748", width=2, dash="dash"),
            annotation_text="Priority Threshold",
            annotation_position="top right",
        )

        fig.update_layout(
            title="<b>Lead Score Distribution</b><br><sup>Intelligent Prioritization: Focus on Green Segment First</sup>",
            xaxis_title="Predicted Conversion Probability",
            yaxis_title="Number of Respondents",
            barmode="stack",
            font=dict(family="Arial, sans-serif", size=13),
            height=500,
            width=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )

        out_path = os.path.join(self.output_dir, "lead_score_distribution.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"  Saved lead score distribution to {out_path}")

    # ------------------------------------------------------------------
    # Chart 4: Channel Effectiveness
    # ------------------------------------------------------------------
    def build_channel_effectiveness(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Grouped bar chart comparing outreach channels."""
        channels = list(metrics.keys())
        open_rates = [metrics[ch]["open_rate"] for ch in channels]
        reply_rates = [metrics[ch]["reply_rate"] for ch in channels]
        cost_per_response = [metrics[ch]["cost_per_response"] for ch in channels]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Engagement Rates by Channel", "Cost per Response"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        # Engagement rates
        fig.add_trace(
            go.Bar(
                x=channels,
                y=open_rates,
                name="Open Rate",
                marker_color="#4299E1",
                text=[f"{v:.1f}%" for v in open_rates],
                textposition="outside",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=channels,
                y=reply_rates,
                name="Reply Rate",
                marker_color="#48BB78",
                text=[f"{v:.1f}%" for v in reply_rates],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        # Cost per response
        fig.add_trace(
            go.Bar(
                x=channels,
                y=cost_per_response,
                name="Cost / Response",
                marker_color="#ED8936",
                text=[f"${v:.2f}" for v in cost_per_response],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="<b>Channel Effectiveness Analysis</b>",
            font=dict(family="Arial, sans-serif", size=13),
            height=500,
            width=1050,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            barmode="group",
        )

        # Recommendation annotation
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.18,
            text=(
                "<b>Best Practice:</b> Use LinkedIn for C-Suite engagement; "
                "deploy WhatsApp for reactivation after 7 days of email silence."
            ),
            showarrow=False,
            font=dict(size=12, color="#2D3748"),
            bgcolor="#F7FAFC",
            bordercolor="#CBD5E0",
            borderwidth=1,
            borderpad=8,
        )

        out_path = os.path.join(self.output_dir, "channel_effectiveness.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"  Saved channel effectiveness to {out_path}")


# ------------------------------------------------------------------
# Data Loading Helpers
# ------------------------------------------------------------------
def load_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all source datasets."""
    respondents = pd.read_csv(os.path.join(base_dir, "data", "raw", "respondents.csv"))
    interactions = pd.read_csv(os.path.join(base_dir, "data", "raw", "interactions.csv"))
    simulation = pd.read_csv(os.path.join(base_dir, "data", "simulation_log.csv"))
    triggers = pd.read_csv(os.path.join(base_dir, "data", "trigger_log.csv"))
    return respondents, interactions, simulation, triggers


def compute_manual_funnel(interactions: pd.DataFrame) -> Tuple[List[int], List[float]]:
    """Derive manual-process funnel from historical interaction data."""
    actions_per_resp = interactions.groupby("respondent_id")["action_type"].apply(set)
    total = actions_per_resp.shape[0]

    contacted = total
    opened = sum(1 for a in actions_per_resp if "opened" in a)
    replied = sum(1 for a in actions_per_resp if "replied" in a)
    scheduled = sum(1 for a in actions_per_resp if "scheduled" in a)
    participated = int(scheduled * 0.70)  # industry-average show-up rate

    counts = [contacted, opened, replied, scheduled, participated]
    rates = [c / contacted * 100.0 for c in counts]
    return counts, rates


def compute_ai_funnel(manual_counts: List[int], lift: float = LIFT_MULTIPLIER) -> List[int]:
    """Project AI-powered funnel using conservative stage lifts."""
    contacted = manual_counts[0]
    # Apply progressive lift: open +30%, reply +40%, schedule × lift, participate +15% show-up
    opened = int(manual_counts[1] * 1.30)
    opened = min(opened, contacted)
    replied = int(manual_counts[2] * 1.40)
    replied = min(replied, opened)
    scheduled = int(manual_counts[3] * lift)
    scheduled = min(scheduled, replied)
    participated = int(scheduled * 0.85)  # AI reminder system reduces no-shows
    participated = min(participated, scheduled)
    return [contacted, opened, replied, scheduled, participated]


def compute_channel_metrics(interactions: pd.DataFrame, respondents: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate open rate, reply rate and cost per response by channel."""
    merged = interactions.merge(respondents[["respondent_id", "preferred_contact"]], on="respondent_id", how="left")

    metrics = {}
    # Map Phone -> WhatsApp for demonstration (no real WhatsApp in synthetic data)
    channel_map = {"Email": "Email", "LinkedIn": "LinkedIn", "Phone": "WhatsApp"}

    for src_label, dst_label in channel_map.items():
        subset = merged[merged["preferred_contact"] == src_label]
        total_resp = subset["respondent_id"].nunique()
        if total_resp == 0:
            continue
        actions = subset.groupby("respondent_id")["action_type"].apply(set)
        opened = sum(1 for a in actions if "opened" in a)
        replied = sum(1 for a in actions if "replied" in a)

        open_rate = opened / total_resp * 100.0
        reply_rate = replied / total_resp * 100.0
        # Approximate cost: Email $0.05, LinkedIn $0.15 (time cost), WhatsApp $0.10
        unit_cost = {"Email": 0.05, "LinkedIn": 0.15, "WhatsApp": 0.10}[dst_label]
        cost_per_response = unit_cost / (reply_rate / 100.0) if reply_rate > 0 else 0.0

        metrics[dst_label] = {
            "open_rate": round(open_rate, 1),
            "reply_rate": round(reply_rate, 1),
            "cost_per_response": round(cost_per_response, 2),
        }

    return metrics


def score_all_respondents(respondents: pd.DataFrame) -> List[float]:
    """Score every respondent using the trained LightGBM model."""
    scorer = LeadScoringModel()
    scorer.load_model()

    scores = []
    for _, row in respondents.iterrows():
        profile = {
            "industry": row["industry"],
            "job_level": row["job_level"],
            "region": row["region"],
            "company_size": row["company_size"],
            "past_participation_count": int(row["past_participation_count"]),
            "preferred_contact": row["preferred_contact"],
            "research_topic_match_score": int(row["research_topic_match_score"]),
            "is_hard_to_reach": int(row["is_hard_to_reach"]),
            "avg_historical_response_time": None,
        }
        scores.append(scorer.predict_lead_score(profile))
    return scores


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("RONIN Digital Research – ROI Dashboard Generation")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nLoading datasets...")
    respondents, interactions, simulation, triggers = load_data(base_dir)
    print(f"  Respondents: {len(respondents)} | Interactions: {len(interactions)}")

    # ------------------------------------------------------------------
    # 2. Compute funnels
    # ------------------------------------------------------------------
    print("\nComputing conversion funnels...")
    manual_counts, manual_rates = compute_manual_funnel(interactions)
    ai_counts = compute_ai_funnel(manual_counts)
    ai_rates = [c / ai_counts[0] * 100.0 for c in ai_counts]
    stages = ["Contact", "Open", "Reply", "Schedule", "Participate"]

    print(f"  Manual funnel:  {manual_counts}")
    print(f"  Manual rates:   {[f'{r:.1f}%' for r in manual_rates]}")
    print(f"  AI funnel:      {ai_counts}")
    print(f"  AI rates:       {[f'{r:.1f}%' for r in ai_rates]}")

    manual_participation_rate = manual_counts[-1] / manual_counts[0]
    ai_participation_rate = ai_counts[-1] / ai_counts[0]
    actual_lift = ai_participation_rate / manual_participation_rate if manual_participation_rate > 0 else 0.0
    print(f"  Participation lift: {actual_lift:.1f}x")

    # ------------------------------------------------------------------
    # 3. ROI calculations
    # ------------------------------------------------------------------
    print("\nRunning ROI calculator...")
    calc = ROICalculator()
    roi_results = calc.project_annual_savings(
        monthly_projects=4,
        avg_leads_per_project=250,
        deal_value=DEAL_VALUE_USD,
        manual_participation_rate=manual_participation_rate,
        ai_participation_rate=ai_participation_rate,
    )

    print(f"  Current Manual Monthly Cost:    ${roi_results['manual_monthly_cost']:,.2f}")
    print(f"  AI-Powered Monthly Cost:        ${roi_results['automation_monthly_cost']:,.2f} ({roi_results['savings_percent']:.1f}% savings)")
    print(f"  Break-even Period:              {roi_results['break_even_days']:.1f} days")
    print(f"  Projected Annual Revenue Uplift: ${roi_results['annual_revenue_uplift_usd']:,.2f}")
    print(f"  Annual Labour Savings:          ${roi_results['annual_labour_savings_usd']:,.2f}")
    print(f"  Total Annual Benefit:           ${roi_results['annual_total_benefit_usd']:,.2f}")
    print(f"  ROI Percentage:                 {roi_results['roi_percentage']:.1f}%")

    hours_saved_monthly = (
        (respondents.shape[0] * MINUTES_PER_LEAD_MANUAL / 60.0)
        - (respondents.shape[0] * REVIEW_RATE * MINUTES_PER_LEAD_MANUAL / 60.0)
    )
    print(f"  Efficiency Gain:                {hours_saved_monthly:.0f} hours/month saved per researcher")

    # ------------------------------------------------------------------
    # 4. Score distribution
    # ------------------------------------------------------------------
    print("\nScoring all respondents...")
    scores = score_all_respondents(respondents)
    print(f"  Mean score: {np.mean(scores):.3f} | High-intent (>0.8): {sum(1 for s in scores if s > 0.8)}")

    # ------------------------------------------------------------------
    # 5. Channel metrics
    # ------------------------------------------------------------------
    print("\nComputing channel effectiveness...")
    channel_metrics = compute_channel_metrics(interactions, respondents)
    for ch, m in channel_metrics.items():
        print(f"  {ch}: Open {m['open_rate']:.1f}%, Reply {m['reply_rate']:.1f}%, Cost/Response ${m['cost_per_response']:.2f}")

    # ------------------------------------------------------------------
    # 6. Generate charts
    # ------------------------------------------------------------------
    print("\nGenerating Plotly charts...")
    dash = DashboardGenerator(os.path.join(base_dir, "dashboard"))
    dash.build_funnel_comparison(manual_counts, ai_counts, stages)

    daily_labour_saving = (roi_results["manual_monthly_cost"] - roi_results["automation_monthly_cost"]) / 30.0
    daily_revenue_uplift = roi_results["annual_revenue_uplift_usd"] / 12.0 / 30.0
    dash.build_roi_timeline(daily_labour_saving, daily_revenue_uplift, roi_results["break_even_days"])

    dash.build_lead_score_distribution(scores)
    dash.build_channel_effectiveness(channel_metrics)

    # ------------------------------------------------------------------
    # 7. Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        **roi_results,
        "conversion_lift_multiplier": f"{actual_lift:.1f}x",
        "manual_funnel_counts": {k: int(v) for k, v in zip(stages, manual_counts)},
        "ai_funnel_counts": {k: int(v) for k, v in zip(stages, ai_counts)},
        "channel_metrics": channel_metrics,
        "generated_at": datetime.now().isoformat(),
    }

    json_path = os.path.join(base_dir, "dashboard", "summary_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary metrics saved to {json_path}")

    print("\nDashboard generation complete.")

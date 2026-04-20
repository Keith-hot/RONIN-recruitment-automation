"""
RONIN Digital Research - LightGBM Lead Scoring Model
Predicts the probability of a respondent engaging in a research project.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import lightgbm as lgb


class LeadScoringModel:
    """
    End-to-end lead scoring pipeline for RONIN Digital Research.
    """

    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.median_response_time = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _get_base_dir(self):
        """Return the absolute path to the project root (parent of src/)."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------------
    # 1. Data Loading & Target Construction
    # ------------------------------------------------------------------
    def load_data(self):
        """Load respondents and interactions data from ../data/raw/."""
        base = self._get_base_dir()
        resp_path = os.path.join(base, "data", "raw", "respondents.csv")
        inter_path = os.path.join(base, "data", "raw", "interactions.csv")

        if not os.path.exists(resp_path):
            raise FileNotFoundError(f"Data file not found: {resp_path}")
        if not os.path.exists(inter_path):
            raise FileNotFoundError(f"Data file not found: {inter_path}")

        df_resp = pd.read_csv(resp_path)
        df_inter = pd.read_csv(inter_path)
        return df_resp, df_inter

    def build_target(self, df_resp, df_inter):
        """
        Build binary target y:
        y = 1 if the respondent has any 'replied' or 'scheduled' action.
        y = 0 otherwise.
        """
        positive_actions = {"replied", "scheduled"}
        # Aggregate unique actions per respondent
        actions_per_resp = (
            df_inter.groupby("respondent_id")["action_type"]
            .apply(lambda x: set(x.unique()))
        )
        y = actions_per_resp.apply(
            lambda acts: 1 if bool(positive_actions & acts) else 0
        )
        # Align with df_resp (respondents with zero interactions get y=0)
        y = y.reindex(df_resp["respondent_id"]).fillna(0).astype(int).values
        return y

    # ------------------------------------------------------------------
    # 2. Feature Engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df_resp, df_inter):
        """
        Create modelling features from raw respondent and interaction data.
        Returns (X, df_full) where df_full retains original columns for analysis.
        """
        df = df_resp.copy()

        # --- One-Hot encode industry ---
        industry_dummies = pd.get_dummies(df["industry"], prefix="industry")
        df = pd.concat([df, industry_dummies], axis=1)

        # --- Ordinal encode job_level ---
        job_level_map = {"C-Suite": 4, "Director": 3, "Manager": 2, "Specialist": 1}
        df["job_level_encoded"] = df["job_level"].map(job_level_map)

        # --- Ordinal encode company_size ---
        company_size_map = {"Enterprise": 3, "Mid-market": 2, "SMB": 1}
        df["company_size_encoded"] = df["company_size"].map(company_size_map)

        # --- Binary flags ---
        df["preferred_contact_email"] = (df["preferred_contact"] == "Email").astype(int)
        df["region_apac"] = (df["region"] == "APAC").astype(int)
        df["has_past_participation"] = (df["past_participation_count"] > 0).astype(int)

        # --- Average historical response time ---
        df_inter["time_to_response_hours"] = pd.to_numeric(
            df_inter["time_to_response_hours"], errors="coerce"
        )
        avg_response = df_inter.groupby("respondent_id")["time_to_response_hours"].mean()
        df["avg_historical_response_time"] = df["respondent_id"].map(avg_response)
        self.median_response_time = df["avg_historical_response_time"].median()
        df["avg_historical_response_time"] = df["avg_historical_response_time"].fillna(
            self.median_response_time
        )

        # --- Assemble feature matrix ---
        dummy_cols = sorted(list(industry_dummies.columns))  # deterministic order
        feature_cols = [
            "is_hard_to_reach",
            "past_participation_count",
            "research_topic_match_score",
            "job_level_encoded",
            "company_size_encoded",
            "preferred_contact_email",
            "region_apac",
            "has_past_participation",
            "avg_historical_response_time",
        ] + dummy_cols

        self.feature_columns = feature_cols
        X = df[feature_cols].copy()
        return X, df

    # ------------------------------------------------------------------
    # 3. Model Training
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Train/test split and fit LightGBM."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            learning_rate=0.05,
            n_estimators=200,
            max_depth=6,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
            verbosity=-1,
        )
        self.model.fit(self.X_train, self.y_train)

    def cross_validate(self, X, y):
        """5-Fold stratified CV reporting AUC and F1."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc")
        f1_scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1")

        print("=" * 50)
        print("5-Fold Cross-Validation Results")
        print("=" * 50)
        print(f"Average AUC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
        print(f"Average F1:  {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        print()

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    def evaluate(self):
        """Evaluate on held-out test set."""
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_prob)
        cm = confusion_matrix(self.y_test, y_pred)

        print("=" * 50)
        print("Test Set Evaluation")
        print("=" * 50)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print()
        print("Confusion Matrix:")
        print("                 Predicted")
        print("                 0      1")
        print(f"Actual    0    {cm[0, 0]:5d}  {cm[0, 1]:5d}")
        print(f"          1    {cm[1, 0]:5d}  {cm[1, 1]:5d}")
        print()

        return y_prob

    # ------------------------------------------------------------------
    # 5. Business Insights
    # ------------------------------------------------------------------
    def business_insights(self, df_full):
        """Print RONIN-specific business insights."""
        test_idx = self.X_test.index
        df_test = df_full.loc[test_idx].copy()
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        df_test["predicted_prob"] = y_prob

        # 高意向线索（概率 > 0.8）的实际转化率
        high_intent_mask = df_test["predicted_prob"] > 0.8
        if high_intent_mask.sum() > 0:
            conv = self.y_test[high_intent_mask].mean() * 100
        else:
            conv = 0.0

        # Hard-to-reach 平均预测概率
        htr_mask = df_test["is_hard_to_reach"] == 1
        if htr_mask.sum() > 0:
            htr_prob = df_test.loc[htr_mask, "predicted_prob"].mean() * 100
        else:
            htr_prob = 0.0

        # 老客户平均预测概率
        ret_mask = df_test["past_participation_count"] > 0
        if ret_mask.sum() > 0:
            ret_prob = df_test.loc[ret_mask, "predicted_prob"].mean() * 100
        else:
            ret_prob = 0.0

        print("=" * 50)
        print("Business Insights")
        print("=" * 50)
        print(f"高意向线索预测（概率 > 0.8）在测试集中的实际转化率：{conv:.1f}%")
        print(f"Hard-to-reach 人群（医生/高管）的平均预测概率：{htr_prob:.1f}%")
        print(f"老客户（past_participation > 0）的平均预测概率：{ret_prob:.1f}%")
        print()

    # ------------------------------------------------------------------
    # 6. Feature Importance Visualization
    # ------------------------------------------------------------------
    def plot_feature_importance(self):
        """Save Plotly Top-10 feature importance chart to dashboard/."""
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=True).tail(10)

        fig = go.Figure(go.Bar(
            x=feat_imp["importance"],
            y=feat_imp["feature"],
            orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            title="RONIN Lead Scoring - Key Conversion Drivers",
            xaxis_title="Feature Importance",
            yaxis_title="Feature",
            template="plotly_white",
            height=500,
            margin=dict(l=150, r=30, t=60, b=40),
        )

        base = self._get_base_dir()
        out_path = os.path.join(base, "dashboard", "feature_importance.html")
        fig.write_html(out_path)
        print(f"Feature importance chart saved to: {out_path}")
        print()

    # ------------------------------------------------------------------
    # 7. Model Persistence
    # ------------------------------------------------------------------
    def save_model(self):
        """Serialize model, feature list, and median response time."""
        base = self._get_base_dir()
        model_path = os.path.join(base, "models", "lgbm_lead_scorer_v1.pkl")
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "median_response_time": self.median_response_time,
        }
        joblib.dump(payload, model_path)
        print(f"Model saved to: {model_path}")
        print()

    def load_model(self):
        """Deserialize model artefacts."""
        base = self._get_base_dir()
        model_path = os.path.join(base, "models", "lgbm_lead_scorer_v1.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.median_response_time = payload["median_response_time"]
        print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # 8. Prediction API
    # ------------------------------------------------------------------
    def predict_lead_score(self, respondent_features_dict):
        """
        Predict engagement probability for a single respondent.

        Parameters
        ----------
        respondent_features_dict : dict
            Must contain keys: industry, job_level, region, company_size,
            past_participation_count, preferred_contact,
            research_topic_match_score, is_hard_to_reach.
            Optional: avg_historical_response_time (defaults to median).

        Returns
        -------
        float
            Probability of positive engagement (0-1).
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded. Call fit() or load_model() first.")

        d = respondent_features_dict
        row = {}

        # Numeric / raw features
        row["is_hard_to_reach"] = int(d.get("is_hard_to_reach", 0))
        row["past_participation_count"] = int(d.get("past_participation_count", 0))
        row["research_topic_match_score"] = float(d.get("research_topic_match_score", 60))

        # Ordinal encodings
        job_map = {"C-Suite": 4, "Director": 3, "Manager": 2, "Specialist": 1}
        row["job_level_encoded"] = job_map.get(d.get("job_level", "Specialist"), 1)

        size_map = {"Enterprise": 3, "Mid-market": 2, "SMB": 1}
        row["company_size_encoded"] = size_map.get(d.get("company_size", "SMB"), 1)

        # Binary flags
        row["preferred_contact_email"] = 1 if d.get("preferred_contact") == "Email" else 0
        row["region_apac"] = 1 if d.get("region") == "APAC" else 0
        row["has_past_participation"] = 1 if row["past_participation_count"] > 0 else 0

        # Response time (fallback to median if missing or None)
        avg_time = d.get("avg_historical_response_time")
        if avg_time is None or (isinstance(avg_time, float) and np.isnan(avg_time)):
            avg_time = self.median_response_time
        row["avg_historical_response_time"] = float(avg_time)

        # Industry one-hot (deterministic order matching training)
        industry = d.get("industry", "Technology")
        for ind in ["Finance", "Healthcare", "Manufacturing", "Retail", "Technology"]:
            row[f"industry_{ind}"] = 1 if industry == ind else 0

        # Align to training column order
        X_pred = pd.DataFrame([row])[self.feature_columns]
        prob = self.model.predict_proba(X_pred)[0, 1]
        return float(prob)


# =====================================================================
# Main execution block
# =====================================================================
if __name__ == "__main__":
    scorer = LeadScoringModel()

    try:
        print("Loading data...")
        df_resp, df_inter = scorer.load_data()

        print("Building target variable...")
        y = scorer.build_target(df_resp, df_inter)

        print("Engineering features...")
        X, df_full = scorer.engineer_features(df_resp, df_inter)

        print("Training LightGBM model...")
        scorer.fit(X, y)

        print("Running cross-validation...")
        scorer.cross_validate(X, y)

        print("Evaluating on test set...")
        scorer.evaluate()

        print("Generating business insights...")
        scorer.business_insights(df_full)

        print("Creating feature importance chart...")
        scorer.plot_feature_importance()

        print("Saving model...")
        scorer.save_model()

        # --- Demonstrate prediction API ---
        print("=" * 50)
        print("Prediction API Demo")
        print("=" * 50)
        scorer.load_model()

        sample_hard_to_reach = {
            "industry": "Healthcare",
            "job_level": "C-Suite",
            "region": "North America",
            "company_size": "Enterprise",
            "past_participation_count": 0,
            "preferred_contact": "Email",
            "research_topic_match_score": 85,
            "is_hard_to_reach": 1,
            "avg_historical_response_time": None,
        }
        prob_htr = scorer.predict_lead_score(sample_hard_to_reach)
        print(f"Hard-to-reach sample probability: {prob_htr:.4f}")

        sample_returning = {
            "industry": "Technology",
            "job_level": "Director",
            "region": "APAC",
            "company_size": "Mid-market",
            "past_participation_count": 3,
            "preferred_contact": "LinkedIn",
            "research_topic_match_score": 72,
            "is_hard_to_reach": 0,
            "avg_historical_response_time": 24.0,
        }
        prob_ret = scorer.predict_lead_score(sample_returning)
        print(f"Returning customer sample probability: {prob_ret:.4f}")
        print()
        print("Pipeline completed successfully.")

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

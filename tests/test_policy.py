from __future__ import annotations

import numpy as np

from src.decision.policy import (
    assign_credit_decision,
    assign_risk_grade,
    calculate_decision_metrics,
)


def test_decision_policy_approve_below_threshold(credit_policy):
    threshold = credit_policy["approve_threshold"]

    decision = assign_credit_decision([threshold - 1e-6], threshold, credit_policy["reject_threshold"])

    assert decision.tolist() == ["APPROVE"]


def test_decision_policy_approve_boundary_is_manual_review(credit_policy):
    threshold = credit_policy["approve_threshold"]

    decision = assign_credit_decision([threshold], threshold, credit_policy["reject_threshold"])

    assert decision.tolist() == ["MANUAL_REVIEW"]


def test_decision_policy_mid_pd_is_manual_review(credit_policy):
    approve = credit_policy["approve_threshold"]
    reject = credit_policy["reject_threshold"]

    decision = assign_credit_decision([(approve + reject) / 2], approve, reject)

    assert decision.tolist() == ["MANUAL_REVIEW"]


def test_decision_policy_reject_boundary_is_inclusive(credit_policy):
    reject = credit_policy["reject_threshold"]

    decision = assign_credit_decision([reject], credit_policy["approve_threshold"], reject)

    assert decision.tolist() == ["REJECT"]


def test_decision_policy_rejects_above_threshold(credit_policy):
    reject = credit_policy["reject_threshold"]

    decision = assign_credit_decision([reject + 1e-6], credit_policy["approve_threshold"], reject)

    assert decision.tolist() == ["REJECT"]


def test_policy_threshold_order_is_valid(credit_policy):
    assert credit_policy["approve_threshold"] < credit_policy["reject_threshold"]


def test_risk_grade_order_does_not_improve_as_pd_increases(credit_policy):
    thresholds = credit_policy["risk_grade_thresholds"]
    labels = tuple(credit_policy["risk_grade_labels"])
    pd_values = np.array([0.0, thresholds[0], thresholds[1], thresholds[2], thresholds[3], 1.0])

    grades = assign_risk_grade(pd_values, thresholds, labels=labels)

    assert set(grades).issubset(set(labels))
    assert grades.iloc[0] == "A"
    assert grades.iloc[-1] == "E"
    assert [labels.index(grade) for grade in grades] == sorted(labels.index(grade) for grade in grades)


def test_captured_default_metrics_distinguish_manual_review_and_reject(credit_policy):
    y_true = np.array([1, 1, 1, 0])
    decisions = np.array(["APPROVE", "MANUAL_REVIEW", "REJECT", "REJECT"])

    metrics = calculate_decision_metrics(y_true, decisions, credit_policy["business_utility"])

    assert metrics["captured_default_rate"] == 2 / 3
    assert metrics["rejected_default_capture_rate"] == 1 / 3

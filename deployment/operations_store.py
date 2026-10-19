import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sklearn.metrics import balanced_accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(
    os.getenv("OPERATIONS_DB_PATH", str(PROJECT_ROOT / "runtime" / "operations.sqlite3"))
)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            predicted_label INTEGER NOT NULL,
            probability REAL,
            threshold REAL
        );
        CREATE TABLE IF NOT EXISTS outcomes (
            prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
            actual_delivered INTEGER NOT NULL CHECK (actual_delivered IN (0, 1)),
            realized_value_eur REAL,
            submitted_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            resolution TEXT,
            resolved_at TEXT
        );
        """
    )
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_predictions(endpoint: str, results: list[dict]) -> None:
    with _connection() as connection:
        connection.executemany(
            """
            INSERT INTO predictions
                (prediction_id, created_at, endpoint, predicted_label, probability, threshold)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result["prediction_id"],
                    _utc_now(),
                    endpoint,
                    result["prediction"],
                    result["on_time_probability"],
                    result["threshold"],
                )
                for result in results
            ],
        )


def record_outcome(
    prediction_id: str, actual_delivered: bool, realized_value_eur: float | None
) -> None:
    with _connection() as connection:
        prediction = connection.execute(
            "SELECT 1 FROM predictions WHERE prediction_id = ?", (prediction_id,)
        ).fetchone()
        if prediction is None:
            raise LookupError("prediction_id was not found")
        try:
            connection.execute(
                """
                INSERT INTO outcomes
                    (prediction_id, actual_delivered, realized_value_eur, submitted_at)
                VALUES (?, ?, ?, ?)
                """,
                (prediction_id, int(actual_delivered), realized_value_eur, _utc_now()),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("outcome feedback was already submitted") from exc


def create_support_ticket(category: str, description: str) -> str:
    ticket_id = str(uuid4())
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO support_tickets (ticket_id, created_at, category, description, status)
            VALUES (?, ?, ?, ?, 'open')
            """,
            (ticket_id, _utc_now(), category, description),
        )
    return ticket_id


def resolve_support_ticket(ticket_id: str, resolution: str) -> None:
    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE support_tickets
            SET status = 'resolved', resolution = ?, resolved_at = ?
            WHERE ticket_id = ? AND status = 'open'
            """,
            (resolution, _utc_now(), ticket_id),
        )
        if cursor.rowcount == 0:
            raise LookupError("open support ticket was not found")


def utilization_summary() -> dict:
    with _connection() as connection:
        prediction_count = connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
        feedback_count = connection.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        outcomes = connection.execute(
            """
            SELECT p.predicted_label, p.probability, o.actual_delivered, o.realized_value_eur
            FROM predictions p JOIN outcomes o USING (prediction_id)
            """
        ).fetchall()
        open_tickets = connection.execute(
            "SELECT COUNT(*) FROM support_tickets WHERE status = 'open'"
        ).fetchone()[0]

    true_delayed = sum(
        row["predicted_label"] == 0 and row["actual_delivered"] == 0 for row in outcomes
    )
    missed_delayed = sum(
        row["predicted_label"] == 1 and row["actual_delivered"] == 0 for row in outcomes
    )
    actual_delayed = true_delayed + missed_delayed
    y_true = [row["actual_delivered"] for row in outcomes]
    y_pred = [row["predicted_label"] for row in outcomes]
    probabilities = [row["probability"] for row in outcomes]
    observed_performance = None
    if outcomes:
        observed_performance = {
            "sample_size": len(outcomes),
            "accuracy": sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
            / len(outcomes),
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_true, y_pred))
                if len(set(y_true)) == 2
                else None
            ),
            "roc_auc": (
                float(roc_auc_score(y_true, probabilities))
                if len(set(y_true)) == 2 and all(value is not None for value in probabilities)
                else None
            ),
            "delayed_recall": true_delayed / actual_delayed if actual_delayed else None,
        }
        observed_performance["retraining_recommended"] = (
            len(outcomes) >= 30
            and (
                observed_performance["roc_auc"] is None
                or observed_performance["roc_auc"] < 0.60
                or observed_performance["delayed_recall"] is None
                or observed_performance["delayed_recall"] < 0.40
            )
        )
        observed_performance["recommendation_reason"] = (
            "At least 30 labeled outcomes are required before automated performance action."
            if len(outcomes) < 30
            else "Retrain/reassess when ROC-AUC < 0.60 or delayed recall < 0.40."
        )
    realized_values = [row["realized_value_eur"] for row in outcomes if row["realized_value_eur"] is not None]
    return {
        "prediction_count": prediction_count,
        "outcome_feedback_count": feedback_count,
        "outcome_coverage": feedback_count / prediction_count if prediction_count else None,
        "observed_delayed_recall": (
            true_delayed / actual_delayed if actual_delayed else None
        ),
        "observed_performance": observed_performance,
        "realized_value_records": len(realized_values),
        "realized_value_eur": sum(realized_values) if realized_values else None,
        "open_support_tickets": open_tickets,
    }

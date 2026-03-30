#!/usr/bin/env python
"""Financial Document Processing Flow -- Hybrid deterministic + agentic pipeline."""

import json
import os
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from crewai import Agent, LLM
from crewai.flow.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback, HumanFeedbackResult

from crewai.flow.persistence import persist, SQLiteFlowPersistence

from fraud_detect_flow_session_5.crews.anomaly_crew.anomaly_crew import AnomalyCrew

BASE_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FileProcessingState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str = ""
    doc_type: str = ""
    confidence: float = 0.0
    file_records: list[dict] = Field(default_factory=list)
    file_columns: list[str] = Field(default_factory=list)
    validation_results: dict = Field(default_factory=dict)
    anomaly_verdict: dict = Field(default_factory=dict)
    final_output: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@persist(SQLiteFlowPersistence())
class FileProcessingFlow(Flow[FileProcessingState]):

    @start()
    def ingest_file(self):
        """Read the CSV file and store as DataFrame in state. Pure Python."""
        print(f"\n[1/5] Ingesting file: {self.state.file_path}")

        full_path = self.state.file_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(BASE_DIR, "data", full_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        df = pd.read_csv(full_path)
        self.state.file_columns = list(df.columns)
        self.state.file_records = df.where(df.notna(), None).to_dict(orient="records")
        print(f"    Loaded {len(self.state.file_records)} rows, {len(self.state.file_columns)} columns")
        print(f"    Columns: {self.state.file_columns}")

    @listen(ingest_file)
    def classify_document(self):
        """Single Agent classifies the document type. Agentic step."""
        print(f"\n[2/5] Classifying document")

        df = pd.DataFrame(self.state.file_records)
        sample_rows = df.head(3).to_json(orient="records", indent=2)
        columns = self.state.file_columns

        classifier = Agent(
            role="Financial Document Classifier",
            goal="Identify the exact type of financial document from its column "
                 "structure and sample data. Return ONLY a JSON object.",
            backstory="You are an expert in financial data formats including "
                      "NACHA/ACH batch files, wire transfer records, and check "
                      "deposit data. You classify documents by examining column "
                      "names and data patterns.",
            llm=LLM(model="openai/gpt-4o-mini", temperature=0.0),
            verbose=True,
        )

        result = classifier.kickoff(
            f"Classify this financial document.\n\n"
            f"Columns: {columns}\n\n"
            f"Sample rows:\n{sample_rows}\n\n"
            f"Return ONLY a JSON object with exactly two fields:\n"
            f'  "doc_type": one of "NACHA", "WIRE", "CHECK"\n'
            f'  "confidence": a float between 0.0 and 1.0\n'
            f"No other text."
        )

        try:
            raw = result.raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            self.state.doc_type = parsed.get("doc_type", "UNKNOWN")
            self.state.confidence = float(parsed.get("confidence", 0.0))
        except (json.JSONDecodeError, ValueError):
            self.state.doc_type = "UNKNOWN"
            self.state.confidence = 0.0

        print(f"    Type: {self.state.doc_type} (confidence: {self.state.confidence})")

    @listen(classify_document)
    def validate_document(self):
        """Apply deterministic validation rules per document type. Pure Python."""
        print(f"\n[3/5] Validating document ({self.state.doc_type})")

        df = pd.DataFrame(self.state.file_records)
        errors = []
        warnings = []

        if self.state.doc_type == "NACHA":
            for idx, row in df.iterrows():
                if pd.isna(row.get("receiving_dfi")) or str(row.get("receiving_dfi")) == "0":
                    errors.append(f"Row {idx}: Invalid receiving DFI routing number")
                if pd.isna(row.get("individual_name")) or str(row.get("individual_name")).strip() == "":
                    errors.append(f"Row {idx}: Missing individual name")
                if pd.isna(row.get("dfi_account_number")) or str(row.get("dfi_account_number")) == "0" * 9:
                    errors.append(f"Row {idx}: Invalid account number (all zeros)")
                amt = float(row.get("amount", 0))
                if amt == 0:
                    errors.append(f"Row {idx}: Zero amount transaction")
                if amt > 1000000:
                    warnings.append(f"Row {idx}: Large transaction amount ({amt})")

        elif self.state.doc_type == "WIRE":
            for idx, row in df.iterrows():
                if pd.isna(row.get("sender_name")) or str(row.get("sender_name")).strip() == "":
                    errors.append(f"Row {idx}: Missing sender name")
                if pd.isna(row.get("purpose_code")) or str(row.get("purpose_code")).strip() == "":
                    errors.append(f"Row {idx}: Missing purpose code")
                amt = float(row.get("amount", 0))
                if amt > 5000000:
                    warnings.append(f"Row {idx}: Wire exceeds $5M threshold ({amt})")
                bene_country = str(row.get("beneficiary_country", ""))
                if bene_country in ("PA", "KY", "VG", "BZ"):
                    warnings.append(f"Row {idx}: High-risk jurisdiction ({bene_country})")

        elif self.state.doc_type == "CHECK":
            for idx, row in df.iterrows():
                if pd.isna(row.get("routing_number")) or str(row.get("routing_number")) == "0":
                    errors.append(f"Row {idx}: Invalid routing number")
                if pd.isna(row.get("bank_name")) or str(row.get("bank_name")).strip() == "":
                    errors.append(f"Row {idx}: Missing bank name")
                if pd.isna(row.get("payee_name")) or str(row.get("payee_name")).strip() == "":
                    errors.append(f"Row {idx}: Missing payee name")
                if str(row.get("endorsement_present")).lower() != "true":
                    errors.append(f"Row {idx}: Missing endorsement")
                maker = str(row.get("maker_name", ""))
                payee = str(row.get("payee_name", ""))
                if maker and payee and maker == payee:
                    warnings.append(f"Row {idx}: Maker and payee are the same ({maker})")
                check_date = str(row.get("check_date", ""))
                if check_date and check_date < "2026-01-01":
                    warnings.append(f"Row {idx}: Stale check date ({check_date})")

        validation_passed = len(errors) == 0

        self.state.validation_results = {
            "validation_passed": validation_passed,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

        print(f"    Passed: {validation_passed}")
        print(f"    Errors: {len(errors)}, Warnings: {len(warnings)}")
        for e in errors:
            print(f"      ERROR: {e}")
        for w in warnings:
            print(f"      WARN:  {w}")

    @listen(validate_document)
    def run_anomaly_crew(self):
        """Run 2-agent crew to analyze data for anomalies and risk. Agentic step."""
        print(f"\n[4/5] Running anomaly detection crew")

        file_summary = json.dumps(self.state.file_records, indent=2)
        validation_json = json.dumps(self.state.validation_results, indent=2)

        result = (
            AnomalyCrew()
            .crew()
            .kickoff(inputs={
                "doc_type": self.state.doc_type,
                "file_data": file_summary,
                "validation_results": validation_json,
            })
        )

        if result.pydantic:
            self.state.anomaly_verdict = result.pydantic.model_dump()
        else:
            try:
                self.state.anomaly_verdict = json.loads(result.raw) if result.raw else {}
            except json.JSONDecodeError:
                self.state.anomaly_verdict = {"recommendation": "MANUAL_REVIEW", "raw": result.raw}

        print(f"    Recommendation: {self.state.anomaly_verdict.get('recommendation', 'UNKNOWN')}")
        print(f"    Risk score: {self.state.anomaly_verdict.get('risk_score', 'N/A')}")

    @listen(run_anomaly_crew)
    @human_feedback(
        message="Review the anomaly detection results. Do you approve, reject, or want revisions?",
        emit=["approved", "rejected", "needs_revision"],
        llm="gpt-4o-mini",
        default_outcome="approved",
    )
    def human_review_step(self):
        """Present anomaly verdict for human review."""
        print(f"\n[5/6] Human review step")
        summary = (
            f"Document: {self.state.doc_type}\n"
            f"Risk Score: {self.state.anomaly_verdict.get('risk_score', 'N/A')}\n"
            f"Recommendation: {self.state.anomaly_verdict.get('recommendation', 'UNKNOWN')}\n"
            f"Anomalies: {json.dumps(self.state.anomaly_verdict.get('anomaly_details', []), indent=2)}"
        )
        print(summary)
        return summary

    @listen("approved")
    def on_approved(self, result: HumanFeedbackResult):
        """Human approved — proceed to final verdict."""
        print(f"\n  Approved by reviewer. Feedback: {result.feedback}")
        self.produce_verdict()

    @listen("rejected")
    def on_rejected(self, result: HumanFeedbackResult):
        """Human rejected the results."""
        print(f"\n  REJECTED by reviewer. Reason: {result.feedback}")
        self.state.anomaly_verdict["recommendation"] = "REJECTED"
        self.state.anomaly_verdict["reviewer_feedback"] = result.feedback
        self.produce_verdict()

    @listen("needs_revision")
    def on_needs_revision(self, result: HumanFeedbackResult):
        """Human requested revisions."""
        print(f"\n  Revision requested. Feedback: {result.feedback}")
        self.state.anomaly_verdict["recommendation"] = "MANUAL_REVIEW"
        self.state.anomaly_verdict["reviewer_feedback"] = result.feedback
        self.produce_verdict()

    def produce_verdict(self):
        """Assemble final output and write to file. Pure Python."""
        print(f"\n[6/6] Producing final verdict")

        self.state.final_output = {
            "flow_id": self.state.id,
            "file_path": self.state.file_path,
            "doc_type": self.state.doc_type,
            "classification_confidence": self.state.confidence,
            "total_records": len(self.state.file_records),
            "validation_passed": self.state.validation_results.get("validation_passed"),
            "error_count": self.state.validation_results.get("error_count"),
            "warning_count": self.state.validation_results.get("warning_count"),
            "errors": self.state.validation_results.get("errors", []),
            "warnings": self.state.validation_results.get("warnings", []),
            "anomalies_detected": self.state.anomaly_verdict.get("anomalies_detected"),
            "risk_score": self.state.anomaly_verdict.get("risk_score"),
            "anomaly_details": self.state.anomaly_verdict.get("anomaly_details", []),
            "recommendation": self.state.anomaly_verdict.get("recommendation"),
        }

        output_path = os.path.join(BASE_DIR, "..", "..", "output", "processing_verdict.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.state.final_output, f, indent=2)

        rec = self.state.anomaly_verdict.get("recommendation", "UNKNOWN")
        print(f"\n{'='*50}")
        print(f"  DOC TYPE:       {self.state.doc_type}")
        print(f"  RECORDS:        {len(self.state.file_records)}")
        print(f"  VALIDATION:     {'PASSED' if self.state.validation_results.get('validation_passed') else 'FAILED'}")
        print(f"  ERRORS:         {self.state.validation_results.get('error_count')}")
        print(f"  WARNINGS:       {self.state.validation_results.get('warning_count')}")
        print(f"  RISK SCORE:     {self.state.anomaly_verdict.get('risk_score')}")
        print(f"  RECOMMENDATION: {rec}")
        print(f"{'='*50}")
        print(f"Output: output/processing_verdict.json")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def kickoff():
    file_path = os.environ.get("FILE_PATH", "nacha_batch_001.csv")
    flow = FileProcessingFlow()
    flow.kickoff(inputs={"file_path": file_path})


def plot():
    FileProcessingFlow().plot()


if __name__ == "__main__":
    kickoff()

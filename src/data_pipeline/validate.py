"""Meridian Customer Intelligence Platform - Pandera Schema & Validation."""

import logging
import pandas as pd
import pandera as pa

logger = logging.getLogger(__name__)

# ── Pandera Schema Definition ────────────────────────────────────────────────
CustomerSchema = pa.DataFrameSchema(
    columns={
        "customer_id": pa.Column(
            str,
            checks=[
                pa.Check(lambda s: s.str.startswith("CUST")),
                pa.Check(lambda s: s.str.len() > 4)
            ],
            nullable=False,
            coerce=True,
            description="Unique customer identifier, must start with CUST."
        ),
        "age": pa.Column(
            int,
            checks=pa.Check.gt(0), # Rule 1: positive age
            nullable=False,
            coerce=True,
            description="Age of the customer, must be positive."
        ),
        "education": pa.Column(
            str,
            checks=pa.Check.isin(["primary", "secondary", "tertiary", "unknown"]), # Rule 2: categorical education
            nullable=False,
            coerce=True,
            description="Education level, restricted to standard categories."
        ),
        "job": pa.Column(
            str,
            checks=pa.Check(lambda s: s.str.strip().str.len() > 0), # Rule 3: non-empty job category
            nullable=False,
            coerce=True,
            description="Job type of the customer."
        ),
        "balance": pa.Column(
            float,
            nullable=False,
            coerce=True,
            description="Bank balance, must be numeric (can be positive or negative)."
        ),
        "duration": pa.Column(
            int,
            checks=pa.Check.ge(0), # Rule 4: non-negative duration
            nullable=False,
            coerce=True,
            description="Last contact duration in seconds, must be non-negative."
        ),
        "complaint": pa.Column(
            str,
            nullable=True,
            coerce=True,
            description="Customer complaint text."
        ),
        "converted": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]), # Rule 6: binary converted flag (target)
            nullable=False,
            coerce=True,
            description="Conversion status (1 if converted, 0 otherwise)."
        )
    },
    strict=False,
    coerce=True
)

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validates the input DataFrame against the CustomerSchema.

    Args:
        df: Input pandas DataFrame containing customer records.

    Returns:
        Validated pandas DataFrame with coerced data types.

    Raises:
        pa.errors.SchemaErrors: If validation fails.
    """
    logger.info("Starting Pandera validation of customer DataFrame...")
    try:
        validated_df = CustomerSchema.validate(df, lazy=True)
        logger.info("Pandera validation passed successfully.")
        return validated_df
    except pa.errors.SchemaErrors as err:
        logger.error("Pandera validation failed with schema errors.")
        # Log detail of errors - use getattr for cross-version Pandera compatibility
        try:
            for failure in err.failure_cases.itertuples():
                col = getattr(failure, "column", getattr(failure, "schema_context", "?"))
                constraint = getattr(failure, "check", "?")
                value = getattr(failure, "failure_case", "?")
                logger.error(f"Column: {col}, Constraint: {constraint}, Failed Value: {value}")
        except Exception:
            logger.error(f"Validation failure details:\n{err.failure_cases.to_string()}")
        raise err

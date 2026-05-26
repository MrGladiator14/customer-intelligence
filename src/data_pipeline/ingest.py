"""Meridian Customer Intelligence Platform - Ingestion Logic."""

import logging
from pathlib import Path
import pandas as pd
from src.data_pipeline.validate import validate_dataframe

logger = logging.getLogger(__name__)

def ingest_csv(filepath: str | Path) -> pd.DataFrame:
    """Reads a CSV file and validates it against the Pandera schema.

    Args:
        filepath: Path to the input CSV file.

    Returns:
        Validated and coerced pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pandera.errors.SchemaErrors: If validation fails.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Inference/Training data file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Ingesting data from {path}...")
    df = pd.read_csv(path)
    
    # Run Pandera validation
    validated_df = validate_dataframe(df)
    return validated_df

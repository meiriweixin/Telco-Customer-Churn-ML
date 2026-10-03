import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for the Telco Customer Churn dataset using
    Great Expectations (1.x API).

    This function implements critical data quality checks that must pass before model
    training. It validates data integrity, business logic constraints, and statistical
    properties that the ML model expects.

    Returns:
        (is_valid, failed_expectations) where failed_expectations is a list of the
        failing expectation type names (e.g. "expect_column_values_to_be_in_set").
    """
    print("🔍 Starting data validation with Great Expectations...")

    # In the raw Telco dataset, TotalCharges is stored as a string with ~11 blank
    # (" ") values for brand-new customers (tenure == 0). Preprocessing coerces it
    # to numeric later in the pipeline, but validation runs on the raw frame first.
    # Coerce a local copy here so the numeric expectations below check the real
    # values instead of failing on the string representation. Blanks become NaN,
    # which GE's value/pair expectations skip by default.
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # === WRAP THE DATAFRAME AS A GE BATCH (ephemeral, in-memory context) ===
    # GE 1.x replaced the old 0.x `ge.dataset.PandasDataset` with a Data Source ->
    # Data Asset -> Batch Definition flow. An ephemeral context keeps everything in
    # memory so nothing is persisted to disk.
    context = gx.get_context(mode="ephemeral")
    batch_definition = (
        context.data_sources.add_pandas("pandas_source")
        .add_dataframe_asset("telco_asset")
        .add_batch_definition_whole_dataframe("telco_batch")
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # === BUILD THE EXPECTATION SUITE ===
    suite = gx.ExpectationSuite(name="telco_churn_suite")

    # --- Schema validation: essential columns must exist ---
    print("   📋 Validating schema and required columns...")
    required_cols = [
        "customerID",
        "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges",
    ]
    for col in required_cols:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    # Customer identifier must not be null (required for business operations)
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="customerID"))

    # --- Business logic validation: categorical value sets ---
    print("   💼 Validating business logic constraints...")
    value_sets = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
    }
    for col, allowed in value_sets.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=col, value_set=allowed)
        )

    # --- Numeric range validation ---
    print("   📊 Validating numeric ranges and business constraints...")
    # Tenure: non-negative and within a reasonable telecom range (~10 years)
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120)
    )
    # Monthly charges: positive and within a reasonable business range
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200)
    )
    # Total charges: non-negative
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0)
    )
    # No missing values in critical numeric features
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))

    # --- Data consistency: cross-column business logic ---
    print("   🔗 Validating data consistency...")
    # TotalCharges should generally be >= MonthlyCharges (allow 5% exceptions for
    # brand-new customers / data entry edge cases).
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,
        )
    )

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = batch.validate(suite)

    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results.results:
        if not r.success:
            cfg = r.expectation_config
            # GE 1.x exposes the expectation type via `.type`
            failed_expectations.append(getattr(cfg, "type", str(cfg)))

    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return bool(results.success), failed_expectations

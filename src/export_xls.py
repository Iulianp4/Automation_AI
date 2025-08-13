import pandas as pd

def export_report(path_xlsx: str, generated_df: pd.DataFrame, mapping_df: pd.DataFrame, metrics_df: pd.DataFrame, manual_df: pd.DataFrame):
    with pd.ExcelWriter(path_xlsx) as writer:
        generated_df.to_excel(writer, sheet_name="generated", index=False)
        mapping_df.to_excel(writer, sheet_name="mapping", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        manual_df.to_excel(writer, sheet_name="manual", index=False)

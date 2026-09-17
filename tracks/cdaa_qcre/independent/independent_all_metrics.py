"""Score all offline compiler proxies, including Qiskit and TKET."""

from pathlib import Path
import independent_metrics as metrics


metrics.COMPILERS = {
    "sabre0330": ("eagle", "heron"),
    "sqgm": ("eagle", "heron"),
    "qiskit141_proxy": ("eagle", "heron"),
    "tket_proxy": ("heron",),
}
metrics.OUT = Path(__file__).resolve().parent / "metrics" / "independent_all_metrics.csv"


if __name__ == "__main__":
    metrics.main()

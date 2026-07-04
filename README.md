# Heat Sink Thermal-AI Surrogate Project

Assessment project for Expert Thermal / XThermal — parameterizing a heat sink
thermal model, generating a synthetic dataset via parameter sweep, and
training ML surrogate models to predict thermal resistance and junction
temperature.

## Project Structure

```
heat_sink_ai_project/
├── data/
│   └── heat_sink_dataset.csv        # Generated in Phase 4
├── src/
│   ├── heat_sink_model.py           # Phase 2: parameterized physics model
│   ├── parameter_sweep.py           # Phase 3-4: sweep + CSV generation
│   ├── train_models.py              # Phase 6-7: ML models + evaluation
│   └── sensitivity_analysis.py      # Phase 8: correlation/sensitivity
├── notebooks/
│   └── exploration.ipynb            # Optional: EDA / plots
├── reports/
│   └── summary_report.md            # Phase 10: final write-up
├── requirements.txt
└── README.md
```

## Milestone Tracker

- [x] Phase 1: Understand existing physics model
- [ ] Phase 2: Parameterize the model (function signature, add R_cond)
- [ ] Phase 3: Build parameter sweep pipeline
- [ ] Phase 4: Generate CSV dataset
- [ ] Phase 5: Exploratory data analysis
- [ ] Phase 6: Train ML models (Linear Regression, Random Forest)
- [ ] Phase 7: Evaluate models (MAE, RMSE, R²)
- [ ] Phase 8: Correlation / sensitivity analysis
- [ ] Phase 9: Conceptual AI questions (Tasks 2-4 of assessment)
- [ ] Phase 10: Summary report

## Known Model Adjustments vs. Original Script

- Added `R_cond` (base conduction resistance) — present in reference PDF's
  resistor network but missing from the original script.
- Corrected `R_jc` from 0.2 to 0.1 °C/W to match the PDF's reference example.
- Refactored global variables into function parameters for sweepability.

Note: trained_models.pkl (~122MB) is excluded from version control 
(exceeds GitHub's file size limit). Re-run train_models.py to 
regenerate it locally.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


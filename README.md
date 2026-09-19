# Deep SSI risk assessment

Streamlit app using the locked six predictor CatBoost model. It calculates `predict_proba`, applies the fixed OOF recalibration `logit(p_cal) = -0.679864 + 1.023649 × logit(p_raw)`, then displays the calibrated probability and risk level. The boundaries are 0.0540609 and 0.3043438 on the calibrated scale.

## Required model files

Place these files in `model_artifacts/` before deployment:

- `final_catboost_6predictor.cbm`
- `final_catboost_6predictor_config.json`

The repository includes these trained files in `model_artifacts/`. To reproduce them from the locked development set, run:

```bash
python ssi_catboost_app_complete.py build --train /path/to/training.xlsx --outdir model_artifacts
```

The build command also writes an audit JSON. Review it before deployment. Rebuilding on different data changes the model.

## Run and deploy

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Push this directory, including `model_artifacts/`, to a GitHub repository. In [Streamlit Community Cloud](https://share.streamlit.io/deploy), select that repository and branch, and set the entrypoint to `streamlit_app.py`.

Enter all six predictors using their development dataset definitions and units. The app does not impute missing inputs and does not show the raw CatBoost probability.

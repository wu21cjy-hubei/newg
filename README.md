# Deep SSI risk assessment

Streamlit app using the locked six predictor CatBoost model. It calculates `predict_proba`, applies the fixed OOF recalibration `logit(p_cal) = -0.679864 + 1.023649 × logit(p_raw)`, then displays the calibrated probability and risk level. The boundaries are 0.0540609 and 0.3043438 on the calibrated scale.

## Required model files

The deployed app reads these files from the repository root:

- `final_catboost_6predictor.cbm`
- `final_catboost_6predictor_config.json`

To reproduce the model from the locked development set, run:

```bash
python ssi_catboost_app_complete.py build --train /path/to/training.xlsx --outdir model_artifacts
```

The build command also writes an audit JSON. Review it before deployment. Rebuilding on different data changes the model.

## Run and deploy

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Push the app files and the three generated model artifacts to the repository root. In [Streamlit Community Cloud](https://share.streamlit.io/deploy), select the repository and `main` branch, set the entrypoint to `streamlit_app.py`, and use Python 3.12.

Live app: https://agkkixvyswcuayw2h7zbn8.streamlit.app/

Enter all six predictors using their development dataset definitions and units. The app does not impute missing inputs and does not show the raw CatBoost probability.

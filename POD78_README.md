# Deep SSI risk assessment — POD7–8

The app accepts Gender and six laboratory variables. It uses the seven predictors present in the supplied research script and datasets, in their original order.

Classification follows the research script: raw CatBoost probability >= 0.11 gives High risk; otherwise Low risk. There is no Intermediate risk category.

For the displayed estimated probability, logistic recalibration is fitted once on training five-fold out-of-fold predictions using the supplied script's calibration routine. This is an app display extension: the research script calculates calibration diagnostics but does not apply recalibration to its predictions. Calibration does not alter the raw-scale classification threshold. The external set is used only for validation, never for model fitting, calibration, or threshold selection.

## Run

```sh
pip install -r requirements.txt
streamlit run pod78_streamlit_app.py
```

For Streamlit Community Cloud, use Python 3.12 and entrypoint `pod78_streamlit_app.py` if these files are in the repository root, or `pod78_app/pod78_streamlit_app.py` if the folder is uploaded intact. Keep the model, config and `pod78_model.py` alongside the entrypoint. A separate deployment can use the same repository as the earlier app.

The deployment does not need patient-level training or validation spreadsheets. Do not upload those datasets to the public repository.

## Rebuild locally

`build_pod78.py` additionally needs the original research dependencies (scikit-learn, imbalanced-learn, matplotlib and openpyxl). Run it with `--source`, `--train` and `--external` pointing to the supplied local research script and datasets. The build saves the model, preprocessing and calibration configuration, and an aggregate validation audit. No patient-level predictions are exported.

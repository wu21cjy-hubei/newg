# Deep SSI risk assessment — POD7–8

The app accepts Gender and six laboratory variables. It uses the seven predictors present in the supplied research script and datasets, in their original order.

Classification uses the locked calibrated probability from `final_CatBoost(pod7-8).py`:

- Low risk: p < 0.03403580844439361
- Intermediate risk: 0.03403580844439361 ≤ p < 0.2871693362170313
- High risk: p ≥ 0.2871693362170313

The displayed probability applies the source's locked OOF calibration: logit(p_cal) = -0.3393279153399011 + 1.457225164081942 × logit(p_raw). The build verifies training OOF compatibility before exporting. The raw threshold 0.1399 is retained for research provenance only; app risk classification uses the two calibrated cutoffs. External data is used only for validation.

## Run

```sh
pip install -r requirements.txt
streamlit run pod78_streamlit_app.py
```

For Streamlit Community Cloud, use Python 3.12 and entrypoint `pod78_streamlit_app.py` if these files are in the repository root, or `pod78_app/pod78_streamlit_app.py` if the folder is uploaded intact. Keep the model, config and `pod78_model.py` alongside the entrypoint. A separate deployment can use the same repository as the earlier app.

The deployment does not need patient-level training or validation spreadsheets. Do not upload those datasets to the public repository.

## Rebuild locally

`build_pod78.py` additionally needs the original research dependencies (scikit-learn, imbalanced-learn, matplotlib and openpyxl). Run it with `--source`, `--train` and `--external` pointing to the supplied local research script and datasets. The build saves the model, preprocessing and calibration configuration, and an aggregate validation audit. No patient-level predictions are exported.

From the project root, rebuild with:

```sh
.venv/bin/python pod78_app/build_pod78.py --source 'final_CatBoost(pod7-8).py' --train '未命名文件夹/训练集_7(1).xlsx' --external '未命名文件夹/外部验证集_7(1).xlsx'
```
